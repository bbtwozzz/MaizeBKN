import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_conv_layer, build_norm_layer)
from mmengine.model import constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmpose.registry import MODELS
from mmpose.utils import get_root_logger
from mmpose.models.backbones.resnet import BasicBlock, Bottleneck
from mmpose.models.backbones.utils import load_checkpoint, channel_shuffle


try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  THOP not available. Install with: pip install thop")


class ModelAnalyzer:

    @staticmethod
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    @staticmethod
    def calculate_model_size(model, unit='MB'):
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_bytes = param_size + buffer_size

        if unit == 'KB':
            return size_bytes / 1024
        elif unit == 'MB':
            return size_bytes / (1024 ** 2)
        elif unit == 'GB':
            return size_bytes / (1024 ** 3)
        else:
            return size_bytes

    @staticmethod
    def calculate_flops(model, input_shape=(1, 3, 416, 624)):
        if not THOP_AVAILABLE:
            return None, "THOP not available"

        try:
            dummy_input = torch.randn(input_shape)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_readable = clever_format([flops], "%.2f")
            params_readable = clever_format([params], "%.2f")
            return flops, flops_readable[0], params, params_readable[0]
        except Exception as e:
            return None, f"Error calculating FLOPs: {str(e)}"

    @staticmethod
    def analyze_model_components(model):
        component_params = {}

        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            component_params[name] = module_params

        return component_params

    @staticmethod
    def format_number(num, precision=2):
        if num >= 1e9:
            return f"{num / 1e9:.{precision}f}G"
        elif num >= 1e6:
            return f"{num / 1e6:.{precision}f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"


class SpatialWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert isinstance(act_cfg, (tuple, list)) and all(isinstance(cfg, dict) for cfg in act_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert isinstance(act_cfg, (tuple, list)) and all(isinstance(cfg, dict) for cfg in act_cfg)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, stride, reduce_ratio, conv_cfg=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]
        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels, ratio=reduce_ratio, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.depthwise_convs = nn.ModuleList([
            ConvModule(channel, channel, kernel_size=3, stride=self.stride, padding=1, groups=channel,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None) for channel in branch_channels
        ])
        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4) for channel in branch_channels
        ])

    def forward(self, x):
        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]
            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]
            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class Stem(nn.Module):
    def __init__(self, in_channels, stem_channels, out_channels, expand_ratio, conv_cfg=None, norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=2, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1, groups=branch_channels,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None),
            ConvModule(branch_channels, inc_channels, kernel_size=1, stride=1, padding=0,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
        )

        self.expand_conv = ConvModule(branch_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                                         groups=mid_channels,
                                         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels, branch_channels if stem_channels == self.out_channels else stem_channels,
            kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))

    def forward(self, x):
        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)
            out = torch.cat((self.branch1(x1), x2), dim=1)
            out = channel_shuffle(out, 2)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class IterativeHead(nn.Module):
    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i], out_channels=self.in_channels[i + 1],
                        kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None, pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i], out_channels=self.in_channels[i],
                        kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None, pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(last_x, size=s.size()[-2:], mode='bilinear', align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s
        return y[::-1]


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'), with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2
        if in_channels != branch_features * 2:
            assert self.stride != 1

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1, groups=in_channels,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None),
                ConvModule(in_channels, branch_features, kernel_size=1, stride=1, padding=0,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(in_channels if (self.stride > 1) else branch_features, branch_features,
                       kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,
                       groups=branch_features,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None),
            ConvModule(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):
        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)
            out = channel_shuffle(out, 2)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class LiteHRModule(nn.Module):
    def __init__(self, num_branches, num_blocks, in_channels, reduce_ratio, module_type,
                 multiscale_output=False, with_fuse=True, conv_cfg=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        self._check_branches(num_branches, in_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels, stride=stride, reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        layers = []
        layers.append(
            ShuffleUnit(self.in_channels[branch_index], self.in_channels[branch_index], stride=stride,
                        conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'),
                        with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(self.in_channels[branch_index], self.in_channels[branch_index], stride=1,
                            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU'),
                            with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1,
                                             padding=0, bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3,
                                                     stride=2, padding=1, groups=in_channels[j], bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1,
                                                     stride=1, padding=0, bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3,
                                                     stride=2, padding=1, groups=in_channels[j], bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=1,
                                                     stride=1, padding=0, bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        fused_feature = self.fuse_layers[i][j](out[j])
                        if fused_feature.shape[-2:] != y.shape[-2:]:
                            fused_feature = F.interpolate(fused_feature, size=y.shape[-2:], mode='bilinear',
                                                          align_corners=False)
                        y += fused_feature
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out

# The MABD proposed in this paper.
class CompactBoundaryDetector(nn.Module):

    def __init__(self, in_channels=32):
        super().__init__()

        self.leaf_sheath_detector = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels, out_channels=12, kernel_size=5,
                stride=1, padding=2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU(),
            nn.Conv2d(12, in_channels, 1),
            nn.Sigmoid()
        )

        self.sheath_mesocotyl_detector = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels, out_channels=12, kernel_size=3,
                stride=1, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU(),
            DepthwiseSeparableConvModule(
                in_channels=12, out_channels=12, kernel_size=(1, 3),
                stride=1, padding=(0, 1), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU(),
            nn.Conv2d(12, in_channels, 1),
            nn.Sigmoid()
        )

        self.mesocotyl_seed_detector = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels, out_channels=12, kernel_size=3,
                stride=1, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU(),
            nn.Conv2d(12, in_channels, 1),
            nn.Sigmoid()
        )

        self.mesocotyl_continuous_detector = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels, out_channels=12, kernel_size=(1, 5),
                stride=1, padding=(0, 2), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU(),
            nn.Conv2d(12, in_channels, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels * 4, out_channels=in_channels, kernel_size=3,
                stride=1, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ),
            nn.ReLU()
        )

    def forward(self, x):

        leaf_sheath_map = self.leaf_sheath_detector(x)
        sheath_mesocotyl_map = self.sheath_mesocotyl_detector(x)
        mesocotyl_seed_map = self.mesocotyl_seed_detector(x)
        mesocotyl_continuous_map = self.mesocotyl_continuous_detector(x)

        enhanced_features = []
        enhanced_features.append(x * (1.0 + 0.3 * leaf_sheath_map))
        enhanced_features.append(x * (1.0 + 0.5 * sheath_mesocotyl_map))
        enhanced_features.append(x * (1.0 + 0.3 * mesocotyl_seed_map))
        enhanced_features.append(x * (1.0 + 0.2 * mesocotyl_continuous_map))

        concatenated = torch.cat(enhanced_features, dim=1)
        fused_features = self.fusion(concatenated)

        boundary_maps = {
            'leaf_sheath': leaf_sheath_map,
            'sheath_mesocotyl': sheath_mesocotyl_map,
            'mesocotyl_seed': mesocotyl_seed_map,
            'mesocotyl_continuous': mesocotyl_continuous_map
        }

        return fused_features, boundary_maps


@MODELS.register_module()
class MaizeBKN(nn.Module):

    def __init__(self, in_channels=3, num_joints=6, conv_cfg=None, norm_cfg=dict(type='BN'),
                 norm_eval=False, with_cp=False, zero_init_residual=False):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.num_joints = num_joints

        # ===== STEM MODULE =====
        self.stem = Stem(
            in_channels=3,
            stem_channels=32,
            out_channels=32,
            expand_ratio=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            with_cp=self.with_cp
        )

        # ===== TRANSITION 0: 32→[32,64] =====
        self.transition0 = nn.ModuleList([
            nn.Sequential(
                build_conv_layer(self.conv_cfg, 32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
                build_norm_layer(self.norm_cfg, 32)[1],
                build_conv_layer(self.conv_cfg, 32, 32, kernel_size=1, stride=1, padding=0, bias=False),
                build_norm_layer(self.norm_cfg, 32)[1],
                nn.ReLU()
            ),
            nn.Sequential(
                build_conv_layer(self.conv_cfg, 32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                build_norm_layer(self.norm_cfg, 32)[1],
                build_conv_layer(self.conv_cfg, 32, 64, kernel_size=1, stride=1, padding=0, bias=False),
                build_norm_layer(self.norm_cfg, 64)[1],
                nn.ReLU()
            )
        ])

        # ===== STAGE 0: 2 modules, 2 branches, [32,64] =====
        self.stage0_module0 = LiteHRModule(
            num_branches=2, num_blocks=2, in_channels=[32, 64], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )
        self.stage0_module1 = LiteHRModule(
            num_branches=2, num_blocks=2, in_channels=[32, 64], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )

        # ===== TRANSITION 1: [32,64]→[32,64,128] =====
        self.transition1 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                build_conv_layer(self.conv_cfg, 64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
                build_norm_layer(self.norm_cfg, 64)[1],
                build_conv_layer(self.conv_cfg, 64, 128, kernel_size=1, stride=1, padding=0, bias=False),
                build_norm_layer(self.norm_cfg, 128)[1],
                nn.ReLU()
            )
        ])

        # ===== STAGE 1: 3 modules, 3 branches, [32,64,128] =====
        self.stage1_module0 = LiteHRModule(
            num_branches=3, num_blocks=2, in_channels=[32, 64, 128], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )
        self.stage1_module1 = LiteHRModule(
            num_branches=3, num_blocks=2, in_channels=[32, 64, 128], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )
        self.stage1_module2 = LiteHRModule(
            num_branches=3, num_blocks=2, in_channels=[32, 64, 128], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )

        # ===== TRANSITION 2: [32,64,128]→[32,64,128,256] =====
        self.transition2 = nn.ModuleList([
            None, None, None,
            nn.Sequential(
                build_conv_layer(self.conv_cfg, 128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
                build_norm_layer(self.norm_cfg, 128)[1],
                build_conv_layer(self.conv_cfg, 128, 256, kernel_size=1, stride=1, padding=0, bias=False),
                build_norm_layer(self.norm_cfg, 256)[1],
                nn.ReLU()
            )
        ])

        # ===== STAGE 2: 2 modules, 4 branches, [32,64,128,256] =====
        self.stage2_module0 = LiteHRModule(
            num_branches=4, num_blocks=2, in_channels=[32, 64, 128, 256], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )
        self.stage2_module1 = LiteHRModule(
            num_branches=4, num_blocks=2, in_channels=[32, 64, 128, 256], reduce_ratio=8,
            module_type='LITE', multiscale_output=True, with_fuse=True,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, with_cp=self.with_cp
        )

        # ===== ITERATIVE HEAD =====
        self.head_layer = IterativeHead(
            in_channels=[32, 64, 128, 256],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )

        # ===== BoundaryDetector =====
        self.boundary_detector = CompactBoundaryDetector(in_channels=32)

        # ===== Final_layer =====
        self.final_layer = nn.Conv2d(
            in_channels=32,
            out_channels=num_joints,
            kernel_size=1, stride=1, padding=0
        )

        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)

        print("✅ MaizeBKN built successfully!")
        
    def get_model_complexity(self, input_shape=(1, 3, 416, 624), verbose=True):
        analyzer = ModelAnalyzer()

        total_params, trainable_params = analyzer.count_parameters(self)

        model_size_mb = analyzer.calculate_model_size(self, unit='MB')

        flops_info = analyzer.calculate_flops(self, input_shape)

        component_params = analyzer.analyze_model_components(self)

        complexity_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'component_params': component_params
        }

        if flops_info[0] is not None:
            complexity_info.update({
                'flops': flops_info[0],
                'flops_readable': flops_info[1],
                'thop_params': flops_info[2],
                'thop_params_readable': flops_info[3]
            })

        if verbose:
            self._print_complexity_info(complexity_info)

        return complexity_info

    def _print_complexity_info(self, info):
        print("\n" + "🔧 " + "=" * 68)
        print("📊 MODEL COMPLEXITY ANALYSIS")
        print("=" * 70)

        print(f"📈 Total Parameters: {ModelAnalyzer.format_number(info['total_params'])}")
        print(f"🎯 Trainable Parameters: {ModelAnalyzer.format_number(info['trainable_params'])}")
        print(f"💾 Model Size: {info['model_size_mb']:.2f} MB")

        if 'flops' in info:
            print(f"⚡ FLOPs: {info['flops_readable']}")
            print(f"🔄 THOP Parameters: {info['thop_params_readable']}")
        else:
            print("⚠️  FLOPs: Not available (install thop: pip install thop)")

        print("\n" + "📋 COMPONENT BREAKDOWN:")
        print("-" * 70)

        total = info['total_params']
        for name, params in info['component_params'].items():
            percentage = (params / total) * 100
            print(f"├── {name:<20}: {ModelAnalyzer.format_number(params):>8} ({percentage:5.2f}%)")

        print("=" * 70 + "\n")

    def forward(self, x, training=True):
        # Stem
        x = self.stem(x)

        # STAGE 0
        x_list = []
        for j in range(2):
            if self.transition0[j]:
                x_list.append(self.transition0[j](x))
            else:
                x_list.append(x)
        x_list = self.stage0_module0(x_list)
        x_list = self.stage0_module1(x_list)

        # STAGE 1
        y_list = []
        for j in range(3):
            if j < len(x_list):
                if self.transition1[j]:
                    y_list.append(self.transition1[j](x_list[j]))
                else:
                    y_list.append(x_list[j])
            else:
                y_list.append(self.transition1[j](x_list[-1]))
        y_list = self.stage1_module0(y_list)
        y_list = self.stage1_module1(y_list)
        y_list = self.stage1_module2(y_list)

        # STAGE 2
        z_list = []
        for j in range(4):
            if j < len(y_list):
                if self.transition2[j]:
                    z_list.append(self.transition2[j](y_list[j]))
                else:
                    z_list.append(y_list[j])
            else:
                z_list.append(self.transition2[j](y_list[-1]))
        z_list = self.stage2_module0(z_list)
        z_list = self.stage2_module1(z_list)

        # Iterative Head
        x = self.head_layer(z_list)

        # === BoundaryDetector ===
        base_features = x[0]  # (B, 32, 104, 156)
        boundary_enhanced_features, boundary_maps = self.boundary_detector(base_features)

        final_prediction = self.final_layer(boundary_enhanced_features)

        if training:
            return final_prediction, {
                'boundary_maps': boundary_maps,
                'enhanced_features': boundary_enhanced_features
            }
        else:
            return final_prediction

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def create_model_with_bd(num_joints=6):
    print("🚀 Creating Lightweight Lite-HRNet + BoundaryDetector...")
    model = MaizeBKN(
        in_channels=3,
        num_joints=num_joints,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        with_cp=False
    )
    return model


def benchmark_model(model, input_shape=(1, 3, 416, 624), num_runs=100):
    import time

    print(f"\n🏃 Running benchmark with {num_runs} iterations...")
    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(input_shape)

        for _ in range(10):
            _ = model(dummy_input, training=False)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(num_runs):
            _ = model(dummy_input, training=False)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time

        print(f"⏱️  Average inference time: {avg_time * 1000:.2f} ms")
        print(f"🚀 Average FPS: {fps:.2f}")


if __name__ == "__main__":
    model = create_model_with_bd(num_joints=6)

    complexity_info = model.get_model_complexity(input_shape=(1, 3, 416, 624))

    total_params_m = complexity_info['total_params'] / 1e6
    flops_g = complexity_info.get('flops', 0) / 1e9 if 'flops' in complexity_info else 0
    print(f"\n📦 Params: {total_params_m:.2f}M")
    print(f"⚡ GFLOPs: {flops_g:.2f} GFLOPs\n")
    # ====================

    print("🧪 Testing forward pass...")
    dummy_input = torch.randn(1, 3, 416, 624)

    output, aux_info = model(dummy_input, training=True)
    print(f"✅ Training output shape: {output.shape}")
    print(f"✅ Boundary maps: {list(aux_info['boundary_maps'].keys())}")
    print(f"✅ Enhanced features shape: {aux_info['enhanced_features'].shape}")

    model.eval()
    with torch.no_grad():
        inference_output = model(dummy_input, training=False)
        print(f"✅ Inference output shape: {inference_output.shape}")

    benchmark_model(model, input_shape=(1, 3, 416, 624), num_runs=50)