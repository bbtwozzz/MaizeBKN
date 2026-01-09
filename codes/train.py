# -*- coding: utf-8 -*-
import json
import os
import datetime
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils import data as torch_data
import numpy as np

import transforms
from final_experiment.models.MaizeBKN import MaizeBKN
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils



BD_CFG = dict(
    r_edge12=6,
    r_disk3=16,
    r_edge45=6,
    r_disk6=16,

    sigma=1.5,

    weights={
        "mask_edge_12": 1.0,
        "mask_disk_3":  1.2,
        "mask_edge_45": 1.0,
        "mask_disk_6":  1.2,
    },

    lambda_bd_final=0.10,
    warmup_epochs=10,

    map_mask_to_bdkey={
        "mask_edge_12": "leaf_sheath",
        "mask_disk_3":  "sheath_mesocotyl",
        "mask_edge_45": "mesocotyl_continuous",
        "mask_disk_6":  "mesocotyl_seed",
    },
)


def _decode_coco_kps(kps_flat: np.ndarray) -> np.ndarray:
    """ [x1,y1,v1, x2,y2,v2, ...] -> (J,3) """
    return np.array(kps_flat, dtype=np.float32).reshape(-1, 3)

def _corridor_from_line_soft(H: int, W: int, p1, p2, r: float, sigma: float) -> np.ndarray:
    x1, y1 = p1; x2, y2 = p2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vx, vy = (x2 - x1), (y2 - y1)
    ux, uy = (xx - x1), (yy - y1)
    t = (ux * vx + uy * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    projx = x1 + t * vx; projy = y1 + t * vy
    dist = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
    if sigma <= 0:
        return (dist <= float(r)).astype(np.float32)
    soft = np.exp(-np.maximum(dist - r, 0.0) ** 2 / (2 * sigma * sigma))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

def _disk_soft(H: int, W: int, c, r: float, sigma: float) -> np.ndarray:
    cx, cy = c
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if sigma <= 0:
        return (dist <= float(r)).astype(np.float32)
    soft = np.exp(-np.maximum(dist - r, 0.0) ** 2 / (2 * sigma * sigma))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

def _build_bd_masks_for_one(
    kps_xyv: np.ndarray,
    out_hw: Tuple[int, int],
    img_hw: Tuple[int, int],
    cfg: Dict
) -> Dict[str, torch.Tensor]:
    H_img, W_img = img_hw
    H_out, W_out = out_hw
    sx = W_out / W_img
    sy = H_out / H_img

    def _pt(ix): return (float(kps_xyv[ix, 0]) * sx, float(kps_xyv[ix, 1]) * sy)
    def _vis(ix): return int(ix) < len(kps_xyv) and kps_xyv[ix, 2] > 0

    r_e12 = cfg["r_edge12"] * 0.5 * (sx + sy)
    r_d3  = cfg["r_disk3"]  * 0.5 * (sx + sy)
    r_e45 = cfg["r_edge45"] * 0.5 * (sx + sy)
    r_d6  = cfg["r_disk6"]  * 0.5 * (sx + sy)
    sigma = cfg["sigma"]

    masks = {}
    if _vis(0) and _vis(1):
        m = _corridor_from_line_soft(H_out, W_out, _pt(0), _pt(1), r=r_e12, sigma=sigma)
        masks["mask_edge_12"] = torch.from_numpy(m).unsqueeze(0)
    if _vis(2):
        m = _disk_soft(H_out, W_out, _pt(2), r=r_d3, sigma=sigma)
        masks["mask_disk_3"] = torch.from_numpy(m).unsqueeze(0)
    if _vis(3) and _vis(4):
        m = _corridor_from_line_soft(H_out, W_out, _pt(3), _pt(4), r=r_e45, sigma=sigma)
        masks["mask_edge_45"] = torch.from_numpy(m).unsqueeze(0)
    if _vis(5):
        m = _disk_soft(H_out, W_out, _pt(5), r=r_d6, sigma=sigma)
        masks["mask_disk_6"] = torch.from_numpy(m).unsqueeze(0)
    return masks

def create_model(num_joints, load_pretrain_weights=False):
    model = MaizeBKN(
        in_channels=3,
        num_joints=num_joints,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False
    )

    if load_pretrain_weights:
        weights_dict = torch.load("./hrnet_w32.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]
            if "final_layer" in k and weights_dict[k].shape[0] != num_joints:
                del weights_dict[k]
            if any(m in k for m in ["boundary_detector", "rl_enhancer"]):
                del weights_dict[k]
        _missing, _unexpected = model.load_state_dict(weights_dict, strict=False)
        if len(_missing) != 0:
            print("missing_keys: ", _missing)
        print("Pretrained weights loaded (partial).")

    return model

def criterion_with_bd(pred, target, epoch: int, max_epoch: int, bd_cfg: Dict):
    if isinstance(pred, tuple):
        pred_hm, aux = pred
    else:
        pred_hm, aux = pred, {}

    if isinstance(target, list) and len(target) > 0 and isinstance(target[0], dict):
        target_hm = torch.stack([t['heatmap'] for t in target], dim=0)  # (B, J, H/4, W/4)
    elif isinstance(target, list):
        target_hm = torch.stack(target, dim=0)
    else:
        target_hm = target
    target_hm = target_hm.to(device=pred_hm.device, dtype=pred_hm.dtype)

    assert pred_hm.shape == target_hm.shape, \
        f"heatmap shape mismatch: pred {pred_hm.shape} vs target {target_hm.shape}"

    main_loss = F.mse_loss(pred_hm, target_hm)

    bd_loss = torch.zeros((), device=pred_hm.device, dtype=pred_hm.dtype)
    num_terms = 0

    if isinstance(pred, tuple) and isinstance(aux, dict) and ('boundary_maps' in aux):
        bd_maps = aux['boundary_maps']  # dict[str] -> (B, Cb, Hb, Wb)
        any_map = next(iter(bd_maps.values()))
        B, Cb, Hb, Wb = any_map.shape

        H_img = target_hm.shape[-2] * 4
        W_img = target_hm.shape[-1] * 4

        kps_list = []
        if isinstance(target, list) and len(target) > 0 and isinstance(target[0], dict):
            for t in target:
                kps = None
                if 'keypoints' in t:
                    arr = np.array(t['keypoints'], dtype=np.float32)
                    kps = _decode_coco_kps(arr) if arr.ndim == 1 else arr
                elif 'kps' in t:
                    arr = np.array(t['kps'], dtype=np.float32)
                    kps = _decode_coco_kps(arr) if arr.ndim == 1 else arr
                kps_list.append(kps)
        else:
            kps_list = [None] * B

        for b in range(B):
            if kps_list[b] is None:
                continue

            masks_b = _build_bd_masks_for_one(
                kps_xyv=kps_list[b],
                out_hw=(Hb, Wb),
                img_hw=(H_img, W_img),
                cfg=bd_cfg
            )

            for k in list(masks_b.keys()):
                m = masks_b[k]
                if m.ndim == 3:
                    m = m.unsqueeze(1)
                elif m.ndim == 4 and m.shape[1] != 1:
                    m = m[:, :1, ...]
                masks_b[k] = m.to(device=pred_hm.device, dtype=pred_hm.dtype)

            for mask_name, bd_key in bd_cfg["map_mask_to_bdkey"].items():
                if (mask_name not in masks_b) or (bd_key not in bd_maps):
                    continue

                pred_map = bd_maps[bd_key][b:b+1].mean(dim=1, keepdim=True)
                pred_map = pred_map.to(device=pred_hm.device, dtype=pred_hm.dtype)

                gt = masks_b[mask_name]

                assert pred_map.shape == gt.shape, \
                    f"BD map shape mismatch for '{bd_key}': pred {pred_map.shape} vs gt {gt.shape}"

                w = float(bd_cfg["weights"].get(mask_name, 1.0))

                bd_loss = bd_loss + w * F.mse_loss(pred_map, gt)
                num_terms += 1

    if num_terms > 0:
        bd_loss = bd_loss / float(num_terms)

    lam_final = float(bd_cfg.get("lambda_bd_final", 0.10))
    warmup_epochs = max(1, int(bd_cfg.get("warmup_epochs", 10)))
    lam = lam_final * float(min(epoch + 1, warmup_epochs)) / float(warmup_epochs)

    total = main_loss + lam * bd_loss

    return total




def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    try:
        torch.cuda.set_per_process_memory_fraction(0.95)
    except Exception:
        pass
    torch.cuda.empty_cache()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    print("=" * 70)
    print("🚀 Training MaizeBKN (with BD pseudo-labels, score-first)")
    print("   - Channels: [32, 64, 128, 256]")
    print("   - Stage1 modules: 3")
    print("   - ✨ WITH CompactBoundaryDetector (supervised, warm-up)")
    print("=" * 70)

    results_file = os.path.join(
        args.output_dir,
        f"results_lw_bd_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    )

    with open(args.keypoints_path, "r", encoding='utf-8') as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)

    if "kps_weights" in person_kps_info:
        kps_weights = np.array(person_kps_info["kps_weights"], dtype=np.float32).reshape((args.num_joints,))
    else:
        kps_weights = np.ones((args.num_joints,), dtype=np.float32)

    if "upper_body_ids" in person_kps_info and "lower_body_ids" in person_kps_info:
        upper_body_ids = person_kps_info["upper_body_ids"]
        lower_body_ids = person_kps_info["lower_body_ids"]
    else:
        print("Warning: 'upper_body_ids' or 'lower_body_ids' not found. Using defaults.")
        upper_body_ids = list(range(3))
        lower_body_ids = list(range(3, args.num_joints))

    if "flip_pairs" in person_kps_info:
        flip_pairs = person_kps_info["flip_pairs"]
    else:
        flip_pairs = None
        print("Warning: 'flip_pairs' not found. Ignoring this parameter.")

    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(scale=(0.65, 1.35), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2,
                                         keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    train_dataset = CocoKeypoint(
        data_root, "train",
        transforms=data_transform["train"],
        fixed_size=args.fixed_size
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=train_dataset.collate_fn
    )

    val_dataset = CocoKeypoint(
        data_root, "val",
        transforms=data_transform["val"],
        fixed_size=args.fixed_size,
        det_json_path=None
    )
    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )

    model = create_model(num_joints=args.num_joints, load_pretrain_weights=args.load_pretrain)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        print("Resume training from epoch {}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch [{epoch}/{args.epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        lam_final = float(BD_CFG.get("lambda_bd_final", 0.10))
        warmup_epochs = max(1, int(BD_CFG.get("warmup_epochs", 10)))
        lam = lam_final * float(min(epoch + 1, warmup_epochs)) / float(warmup_epochs)
        print(f"λ_bd (warm-up): {lam:.4f} (final {lam_final}, warmup {warmup_epochs} epochs)")
        print(f"{'=' * 70}")

        warmup_flag = (epoch == 0)

        def _criterion(pred, tgt):
            return criterion_with_bd(pred, tgt, epoch=epoch, max_epoch=args.epochs, bd_cfg=BD_CFG)

        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_loader,
            device=device, epoch=epoch,
            print_freq=100, warmup=warmup_flag,
            scaler=scaler,
            criterion_func=_criterion
        )

        train_loss.append(mean_loss.item())
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rate.append(current_lr)

        coco_info = utils.evaluate(
            model, val_loader, device=device,
            flip=True, flip_pairs=person_kps_info.get("flip_pairs")
        )

        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])
        print(f"Validation Results:\n   AP @0.5:0.95 = {coco_info[0]:.4f}\n   AP @0.5     = {coco_info[1]:.4f}\n   AR @0.5:0.95 = {coco_info[-1]:.4f}")

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if scaler is not None:
            save_files["scaler"] = scaler.state_dict()

        save_path = os.path.join(args.output_dir, f"model-{epoch}.pth")
        torch.save(save_files, save_path)

        if len(val_map) == 0 or coco_info[1] >= max(val_map):
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(save_files, best_path)
            print(f"   ✅ Best model saved! (AP@0.5: {coco_info[1]:.4f})")

    print("\n" + "=" * 70)
    print("🎉 Training completed!")
    print("=" * 70)

    if len(train_loss) != 0 and len(learning_rate) != 0:
        try:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)
        except Exception:
            pass

    if len(val_map) != 0:
        try:
            from plot_curve import plot_map
            plot_map(val_map)
        except Exception:
            pass
        print(f"Best AP@0.5: {max(val_map):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train MaizeBKN (with BD pseudo-label supervision, score-first)'
    )

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path',
                        default=r"E:\maize_dataset\images",
                        help='dataset root path')
    parser.add_argument('--keypoints-path',
                        default=r"E:\maize_dataset\images\annotations\train_keypoints.json",
                        type=str, help='keypoints annotation json path')
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[416, 624], nargs='+', type=int,
                        help='input size')
    parser.add_argument('--num-joints', default=6, type=int,
                        help='number of keypoints')

    parser.add_argument('--output-dir',
                        default=r'E:\zgxf\MaizeBKN\final_experiment\train_outputs\MaizeBKN',
                        help='path where to save checkpoints')

    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--lr', default=0.0003, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size when training')
    parser.add_argument('--amp', default=True,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument('--load-pretrain', action='store_true',
                        help='whether to load pretrained weights')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Training Configuration (LW + BD + Pseudo-labels, score-first):")
    print("=" * 70)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 70 + "\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    main(args)
