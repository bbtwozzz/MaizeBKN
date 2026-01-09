# -*- coding: utf-8 -*-
import os
import json
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, List, Any
from final_experiment.models.MaizeBKN import MaizeBKN
from draw_utils import draw_keypoints
import transforms
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils

VIS_FIGSIZE = (12, 6)
VIS_DPI     = 380
OVERLAY_ALPHA = 0.5
OVERLAY_SCALE = 2.0

def load_ground_truth_annotations(annotation_file):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in id_to_filename:
            filename = id_to_filename[image_id]
            kf = ann['keypoints']
            keypoints, vis = [], []
            for i in range(0, len(kf), 3):
                x, y, v = kf[i], kf[i+1], kf[i+2]
                keypoints.append([x, y]); vis.append(v)
            annotations[filename] = {
                'keypoints': keypoints,          # (J,2)
                'visibilities': vis,             # (J,)
                'bbox': ann.get('bbox', None),
                'num_keypoints': ann.get('num_keypoints', len(keypoints))
            }
    return annotations

def _corridor_from_line_soft(H: int, W: int, p1, p2, r: float, sigma: float) -> np.ndarray:
    x1, y1 = p1; x2, y2 = p2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vx, vy = (x2-x1), (y2-y1)
    ux, uy = (xx-x1), (yy-y1)
    t = (ux*vx + uy*vy) / (vx*vx + vy*vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    projx = x1 + t*vx; projy = y1 + t*vy
    dist = np.sqrt((xx-projx)**2 + (yy-projy)**2)
    if sigma <= 0:
        return (dist <= float(r)).astype(np.float32)
    soft = np.exp(-np.maximum(dist - r, 0.0)**2 / (2*sigma*sigma))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

def _disk_soft(H: int, W: int, c, r: float, sigma: float) -> np.ndarray:
    cx, cy = c
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    if sigma <= 0:
        return (dist <= float(r)).astype(np.float32)
    soft = np.exp(-np.maximum(dist - r, 0.0)**2 / (2*sigma*sigma))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

def _build_bd_masks_for_one(kps_xyv: np.ndarray, out_hw: Tuple[int, int],
                            img_hw: Tuple[int, int], cfg: Dict) -> Dict[str, torch.Tensor]:
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

BD_CFG = dict(
    r_edge12=6, r_disk3=16, r_edge45=6, r_disk6=16,
    sigma=1.5,
    weights={"mask_edge_12": 1.0, "mask_disk_3": 1.2, "mask_edge_45": 1.0, "mask_disk_6": 1.2},
    map_mask_to_bdkey={
        "mask_edge_12": "leaf_sheath",
        "mask_disk_3":  "sheath_mesocotyl",
        "mask_edge_45": "mesocotyl_continuous",
        "mask_disk_6":  "mesocotyl_seed",
    },
)

def calculate_pck(pred_keypoints, gt_keypoints, visibilities=None, threshold_ratios=[0.05, 0.1, 0.2],
                  normalize_factor=None):
    if visibilities is not None:
        valid = np.array(visibilities) >= 1
        if not np.any(valid):
            return {f'PCK@{t}': 0.0 for t in threshold_ratios}, np.array([])
        pred_keypoints = pred_keypoints[valid]
        gt_keypoints = gt_keypoints[valid]

    if normalize_factor is None:
        max_range = max(np.max(gt_keypoints[:, 0]) - np.min(gt_keypoints[:, 0]),
                        np.max(gt_keypoints[:, 1]) - np.min(gt_keypoints[:, 1]))
        normalize_factor = max_range if max_range > 0 else 1.0

    d = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=1))
    nd = d / normalize_factor
    res = {}
    for thr in threshold_ratios:
        res[f'PCK@{thr}'] = float(np.mean(nd <= thr)) if len(nd) > 0 else 0.0
    return res, nd


def invert_affine_2x3(M: np.ndarray) -> np.ndarray:
    assert M.shape == (2, 3)
    A = M[:, :2]; t = M[:, 2:3]
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    return np.hstack([A_inv, t_inv]).astype(np.float32)

def apply_affine_to_points(xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    xy1 = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)
    return (M @ xy1.T).T


def overlay_heatmap_on_image(img_rgb_uint8: np.ndarray, heatmap_float01: np.ndarray, alpha: float = OVERLAY_ALPHA):
    H, W = img_rgb_uint8.shape[:2]
    hm = heatmap_float01
    if hm.shape[0] != H or hm.shape[1] != W:
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0.0)
    return blend


def create_model_with_bd(num_joints):
    return MaizeBKN(
        in_channels=3, num_joints=num_joints,
        conv_cfg=None, norm_cfg=dict(type='BN'),
        norm_eval=False, with_cp=False, zero_init_residual=False
    )

def load_model_with_bd(model_path, num_joints=6, device='cpu'):
    print(f"📥 load+model: {model_path}")
    model = create_model_with_bd(num_joints)
    ckpt = torch.load(model_path, map_location=device)
    if "model" in ckpt:
        weights = ckpt["model"]; epoch = ckpt.get('epoch', 'unknown')
    else:
        weights = ckpt; epoch = 'unknown'
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:   print(f"⚠️ Missing key {len(missing)}: {missing[:8]}")
    if unexpected:print(f"⚠️ Unexpected key {len(unexpected)}: {unexpected[:8]}")
    print(f"✅ Model loaded successfully (epoch={epoch})")
    print(f"📊 Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model

def create_test_dataloader(test_json_path, images_dir, fixed_size=[416, 624], batch_size=16):
    print(f"📊 Make test: {test_json_path} @ {images_dir}, size={fixed_size}")
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(test_json_path)

    test_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = CocoKeypoint(root=images_dir, dataset="test",
                      transforms=test_transform, fixed_size=fixed_size, det_json_path=None)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    use_persistent = nw > 0

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=nw, persistent_workers=use_persistent,
        collate_fn=ds.collate_fn
    )
    print(f"✅ dataloader ok, N={len(ds)}, batches={len(loader)}")
    return loader, ds

def _safe_num(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip().upper() in ('N/A', 'NA', 'NONE', ''):
            return None
        return float(x)
    except Exception:
        return None

def summarize_from_detailed(detailed_results: List[List[Any]]) -> Dict[str, Any]:
    if not detailed_results:
        return {}

    per_point_vals = [[] for _ in range(6)]
    avg_scores = []
    pck05_vals, pck10_vals, pck20_vals, nme_vals = [], [], [], []

    for row in detailed_results:
        for j in range(6):
            v = _safe_num(row[1 + j])
            if v is not None: per_point_vals[j].append(v)
        v_avg = _safe_num(row[7])
        if v_avg is not None: avg_scores.append(v_avg)

        v05 = _safe_num(row[8]);  v10 = _safe_num(row[9])
        v20 = _safe_num(row[10]); vn  = _safe_num(row[11])
        if v05 is not None: pck05_vals.append(v05)
        if v10 is not None: pck10_vals.append(v10)
        if v20 is not None: pck20_vals.append(v20)
        if vn  is not None: nme_vals.append(vn)

    per_point_mean = [ (float(np.mean(v)) if len(v)>0 else None) for v in per_point_vals ]
    total_avg_score = float(np.mean(avg_scores)) if len(avg_scores) > 0 else None
    pck05_mean = float(np.mean(pck05_vals)) if len(pck05_vals) > 0 else None
    pck10_mean = float(np.mean(pck10_vals)) if len(pck10_vals) > 0 else None
    pck20_mean = float(np.mean(pck20_vals)) if len(pck20_vals) > 0 else None
    nme_mean   = float(np.mean(nme_vals))   if len(nme_vals)   > 0 else None

    if all(v is not None for v in per_point_mean):
        best_idx  = int(np.argmax(per_point_mean))
        worst_idx = int(np.argmin(per_point_mean))
    else:
        best_idx = worst_idx = None

    return dict(
        avg_score=total_avg_score,
        per_point_mean=per_point_mean,
        num_samples=len(avg_scores),
        pck05=pck05_mean, pck10=pck10_mean, pck20=pck20_mean, nme=nme_mean,
        best_idx=best_idx, worst_idx=worst_idx
    )

def save_unified_results(output_folder, timestamp, coco_results, detailed_results,
                         model_name="Lightweight-LiteHRNet-BD",
                         bd_diag_summary: Optional[Dict[str, float]] = None,
                         bd_missing_counts: Optional[Dict[str, int]] = None,
                         total_bd_visual: int = 0):
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, f'evaluation_results_{timestamp}.csv')
    headers = ['Image_Name', 'Point_1', 'Point_2', 'Point_3', 'Point_4', 'Point_5', 'Point_6',
               'Average_Score', 'PCK@0.05', 'PCK@0.1', 'PCK@0.2', 'NME']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(headers)
        for row in detailed_results: w.writerow(row)

    stat = summarize_from_detailed(detailed_results)
    def _fmt(v, pct=False):
        if v is None: return "N/A"
        return f"{v:.4f}" if not pct else f"{v*100:.2f}%"

    summary_path = os.path.join(output_folder, f'evaluation_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("🌽 MaizeBKN report\n")
        f.write("="*80 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if coco_results:
            f.write("📊 COCO indicators (AP/AR):\n")
            f.write("-"*50 + "\n")
            names = ["AP[0.50:0.95]","AP50","AP75","AP(M)","AP(L)",
                     "AR[0.50:0.95]","AR50","AR75","AR(M)","AR(L)"]
            for n, v in zip(names, coco_results):
                f.write(f"{n:<42}: {v:.4f}\n")
            f.write("\n")

        f.write("📊 Keypoint test score statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"🎯 All_Score_avg: {_fmt(stat.get('avg_score'))} ({_fmt(stat.get('avg_score'), pct=True)})\n")
        f.write(f"   (Based on {stat.get('num_samples','N/A')} samples)\n")
        ppm = stat.get('per_point_mean', [None]*6)
        for i, val in enumerate(ppm, start=1):
            f.write(f"   Point {i} avg_score: {_fmt(val)} ({_fmt(val, pct=True)}) [samples: {stat.get('num_samples','N/A')}]\n")
        bi = stat.get('best_idx'); wi = stat.get('worst_idx')
        if isinstance(bi, int):
            f.write(f" Max score: Point_{bi+1} ({_fmt(ppm[bi])})\n")
        if isinstance(wi, int):
            f.write(f" Minimum score: Point_{wi+1} ({_fmt(ppm[wi])})\n")
        f.write("\n")

        f.write("📊 PCK & NME evaluation:\n")
        f.write("-"*50 + "\n")
        f.write(f" PCK@0.05: {_fmt(stat.get('pck05'))} ({_fmt(stat.get('pck05'), pct=True)})\n")
        f.write(f" PCK@0.1:  {_fmt(stat.get('pck10'))} ({_fmt(stat.get('pck10'), pct=True)})\n")
        f.write(f" PCK@0.2:  {_fmt(stat.get('pck20'))} ({_fmt(stat.get('pck20'), pct=True)})\n")
        f.write(f" NME:      {_fmt(stat.get('nme'))} ({_fmt(stat.get('nme'), pct=True)})\n\n")

        ap = coco_results[0] if coco_results else None
 

        if bd_diag_summary:
            f.write("🧪 BD Effectiveness\n")
            f.write("-"*50 + "\n")
            for k, v in bd_diag_summary.items():
                f.write(f"{k:<35}: {v:.4f}\n")
            f.write("\n")
        if bd_missing_counts is not None:
            f.write("📋 BD missing\n")
            f.write("-"*50 + "\n")
            f.write(f"Number of samples: {total_bd_visual}\n")
            for k, v in bd_missing_counts.items():
                f.write(f"{k:<28}: {v}\n")
            f.write("\n")

        f.write(f"CSV: {csv_path}\n")
        f.write(f"📄 Time: {timestamp}\n")

    print(f"📄 CSV: {csv_path}\n📄 report: {summary_path}")
    return csv_path, summary_path

def unified_evaluation(
        model_path,
        test_json_path,
        images_dir,
        output_folder="evaluation",
        device='cuda:0',
        fixed_size=[416, 624],
        batch_size=16,
        num_joints=6,
        flip_test=False,
        save_visualizations=True,
        save_bd_maps=True,
        bd_visual_limit=30,
        bd_visual_seed: Optional[int] = 42,
        eval_bd_supervision=True
):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("🔧 device:", device)

    BD_CFG["img_hw"] = (fixed_size[0], fixed_size[1])

    model = load_model_with_bd(model_path, num_joints=num_joints, device=device)
    model.to(device); model.eval()

    test_loader, test_dataset = create_test_dataloader(test_json_path, images_dir, fixed_size, batch_size)

    gt_ann = load_ground_truth_annotations(test_json_path)
    print(f"📋 GT entry: {len(gt_ann)}")

    with torch.no_grad():
        coco_results = utils.evaluate(model, test_loader, device=device,
                                      flip=flip_test, flip_pairs=None)

    print("\n🔍 Figure by Figure Evaluation...")
    detailed_results = []
    single_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_names = [info['file_name'] for info in test_dataset.coco.imgs.values()]

    if bd_visual_seed is not None:
        random.seed(bd_visual_seed)
    selected_for_bd = set(random.sample(img_names, min(int(bd_visual_limit), len(img_names))))

    processed = 0

    bd_metrics_acc = {
        "Dice_leaf_sheath": [], "BCE_leaf_sheath": [],
        "Dice_sheath_mesocotyl": [], "BCE_sheath_mesocotyl": [],
        "Dice_mesocotyl_continuous": [], "BCE_mesocotyl_continuous": [],
        "Dice_mesocotyl_seed": [], "BCE_mesocotyl_seed": [],
    }
    bd_missing_counts = {
        "leaf_sheath": 0, "sheath_mesocotyl": 0, "mesocotyl_continuous": 0, "mesocotyl_seed": 0
    }
    total_bd_visual = 0

    for img_name in img_names:
        path = os.path.join(images_dir, img_name)
        if not os.path.exists(path): continue

        try:
            # read
            arr = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None: continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # SAME transform as eval
            img_tensor, target = single_transform(img, {"box": [0, 0, img.shape[1]-1, img.shape[0]-1]})
            img_tensor = img_tensor.unsqueeze(0).to(device)

            if "forward_trans" in target:
                forward_M = np.array(target["forward_trans"], dtype=np.float32)
            elif "reverse_trans" in target:
                forward_M = invert_affine_2x3(np.array(target["reverse_trans"], dtype=np.float32))
            else:
                sx = fixed_size[1] / img.shape[1]
                sy = fixed_size[0] / img.shape[0]
                forward_M = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)

            img_fixed = cv2.warpAffine(img, forward_M, (fixed_size[1], fixed_size[0]),
                                       flags=cv2.INTER_LINEAR)

            with torch.inference_mode():
                out = model(img_tensor, training=False)
                if flip_test:
                    flip_tensor = transforms.flip_images(img_tensor)
                    out_flip = model(flip_tensor, training=False)
                    out_flip = torch.squeeze(transforms.flip_back(out_flip, None))
                    out_flip[..., 1:] = out_flip.clone()[..., 0:-1]
                    out = (out + out_flip) * 0.5
                keypoints, scores = transforms.get_final_preds(out, [target["reverse_trans"]], True)
                keypoints = np.squeeze(keypoints); scores = np.squeeze(scores)

            avg_score = float(np.mean(scores))
            row = [
                img_name,
                round(float(scores[0]), 4), round(float(scores[1]), 4), round(float(scores[2]), 4),
                round(float(scores[3]), 4), round(float(scores[4]), 4), round(float(scores[5]), 4),
                round(avg_score, 4)
            ]

            # PCK/NME
            if img_name in gt_ann:
                gt = gt_ann[img_name]
                vis = gt.get('visibilities', None)
                if gt.get('bbox') is not None:
                    _, _, bw, bh = gt['bbox']
                    norm = max(bw, bh) if max(bw, bh) > 0 else None
                else:
                    norm = None
                pck, nme = calculate_pck(keypoints, np.array(gt['keypoints']), visibilities=vis,
                                         threshold_ratios=[0.05, 0.1, 0.2], normalize_factor=norm)
                if len(nme) > 0:
                    row.extend([
                        round(float(pck['PCK@0.05']), 4),
                        round(float(pck['PCK@0.1']), 4),
                        round(float(pck['PCK@0.2']), 4),
                        round(float(np.mean(nme)), 4)
                    ])
                else:
                    row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            else:
                row.extend(['N/A', 'N/A', 'N/A', 'N/A'])

            detailed_results.append(row)

            if (save_bd_maps or eval_bd_supervision) and (img_name in selected_for_bd):
                with torch.no_grad():
                    pred_with_aux = model(img_tensor, training=True)  # return (heatmap, aux)
                _, aux = pred_with_aux
                bd_maps = aux['boundary_maps']  # dict[name] -> (B,C,Hb,Wb)
                any_map = next(iter(bd_maps.values()))
                Hb, Wb = int(any_map.shape[-2]), int(any_map.shape[-1])

                masks = {}
                if img_name in gt_ann:
                    gt_xy = np.array(gt_ann[img_name]['keypoints'], dtype=np.float32)           # (J,2)
                    gt_v = np.array(gt_ann[img_name]['visibilities'], dtype=np.float32).reshape(-1, 1)  # (J,1)
                    xy_fixed = apply_affine_to_points(gt_xy, forward_M)                         # (J,2)
                    kps_xyv_fixed = np.concatenate([xy_fixed, gt_v], axis=1)                    # (J,3)
                    masks = _build_bd_masks_for_one(
                        kps_xyv=kps_xyv_fixed,
                        out_hw=(Hb, Wb),
                        img_hw=(fixed_size[0], fixed_size[1]),
                        cfg=BD_CFG
                    )

                vis_dir = os.path.join(output_folder, "boundary_maps_aligned")
                os.makedirs(vis_dir, exist_ok=True)

                name_map = {
                    "leaf_sheath": ("Dice_leaf_sheath", "BCE_leaf_sheath"),
                    "sheath_mesocotyl": ("Dice_sheath_mesocotyl", "BCE_sheath_mesocotyl"),
                    "mesocotyl_continuous": ("Dice_mesocotyl_continuous", "BCE_mesocotyl_continuous"),
                    "mesocotyl_seed": ("Dice_mesocotyl_seed", "BCE_mesocotyl_seed"),
                }

                for bd_key in ["leaf_sheath", "sheath_mesocotyl", "mesocotyl_continuous", "mesocotyl_seed"]:
                    pred_map = bd_maps[bd_key].mean(dim=1, keepdim=True)[0]  # (1,Hb,Wb)
                    mm = pred_map[0].detach().cpu().numpy()
                    mm = (mm - mm.min()) / (mm.ptp() + 1e-6)

                    # GT mask
                    mask_name = None
                    for k, v in BD_CFG["map_mask_to_bdkey"].items():
                        if v == bd_key: mask_name = k; break
                    gt_mask = masks.get(mask_name, None)

                    if eval_bd_supervision:
                        if gt_mask is not None:
                            gt_mask = gt_mask.to(device=device, dtype=torch.float32)  # (1,Hb,Wb)
                            gm_np = gt_mask[0].detach().cpu().numpy()
                            inter = (pred_map * gt_mask).sum()
                            dice = (2*inter + 1e-6) / (pred_map.sum() + gt_mask.sum() + 1e-6)
                            bce = torch.nn.functional.binary_cross_entropy(pred_map, gt_mask)
                            dk, bk = name_map[bd_key]
                            bd_metrics_acc[dk].append(float(dice.detach().cpu()))
                            bd_metrics_acc[bk].append(float(bce.detach().cpu()))
                        else:
                            gm_np = np.zeros_like(mm)
                            bd_missing_counts[bd_key] += 1

                    plt.figure(figsize=VIS_FIGSIZE)
                    plt.subplot(1,2,1); plt.imshow(mm, cmap='jet'); plt.axis('off'); plt.title(f"pred {bd_key}")
                    if gt_mask is not None:
                        plt.subplot(1,2,2); plt.imshow(gm_np, cmap='jet'); plt.axis('off'); plt.title(mask_name)
                    else:
                        plt.subplot(1,2,2); plt.imshow(np.zeros_like(mm), cmap='jet'); plt.axis('off'); plt.title("no GT")
                    outp = os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}_{bd_key}.png")
                    plt.tight_layout(pad=0.15); plt.savefig(outp, dpi=VIS_DPI); plt.close()

                    overlay_pred = overlay_heatmap_on_image(img_fixed.copy(), mm, alpha=OVERLAY_ALPHA)
                    if gt_mask is not None:
                        overlay_gt = overlay_heatmap_on_image(img_fixed.copy(), gm_np, alpha=OVERLAY_ALPHA)
                    else:
                        overlay_gt = img_fixed.copy()

                    if OVERLAY_SCALE and OVERLAY_SCALE != 1.0:
                        new_w = int(round(overlay_pred.shape[1] * OVERLAY_SCALE))
                        new_h = int(round(overlay_pred.shape[0] * OVERLAY_SCALE))
                        overlay_pred = cv2.resize(overlay_pred, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        overlay_gt   = cv2.resize(overlay_gt,   (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                    plt.figure(figsize=VIS_FIGSIZE)
                    plt.subplot(1,2,1); plt.imshow(overlay_pred); plt.axis('off'); plt.title(f"overlay pred {bd_key}")
                    if gt_mask is not None:
                        plt.subplot(1,2,2); plt.imshow(overlay_gt); plt.axis('off'); plt.title(f"overlay {mask_name}")
                    else:
                        plt.subplot(1,2,2); plt.imshow(overlay_gt); plt.axis('off'); plt.title("overlay no GT")
                    outp_overlay = os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}_{bd_key}_overlay.png")
                    plt.tight_layout(pad=0.15); plt.savefig(outp_overlay, dpi=VIS_DPI); plt.close()

                total_bd_visual += 1

            if save_visualizations:
                plot_img = draw_keypoints(img, keypoints, scores, thresh=0.0001, r=15)
                vis_path = os.path.join(output_folder, f"vis_{img_name}")
                if hasattr(plot_img, 'save'):
                    plot_img.save(vis_path, quality=100, optimize=False)
                else:
                    if len(plot_img.shape) == 3:
                        plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
                    else:
                        plot_img_bgr = plot_img
                    cv2.imwrite(vis_path, plot_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

            processed += 1
            if processed % 50 == 0:
                print(f" Processed {processed}/{len(img_names)}")

        except Exception as e:
            print(f"Processing  {img_name} error: {e}")
            continue

    print(f"✅ Finish，N={processed}")

    bd_diag_summary = None
    if eval_bd_supervision and processed > 0:
        bd_diag_summary = {}
        for k, arr in bd_metrics_acc.items():
            if len(arr) > 0:
                bd_diag_summary[f"{k} (mean)"] = float(np.mean(arr))

    csv_path, summary_path = save_unified_results(
        output_folder, timestamp, coco_results, detailed_results,
        "Lightweight-LiteHRNet-BD (Aligned Pseudo Labels)",
        bd_diag_summary=bd_diag_summary,
        bd_missing_counts=bd_missing_counts if len(selected_for_bd) > 0 else None,
        total_bd_visual=len(selected_for_bd)
    )

    stat = summarize_from_detailed(detailed_results)
    if stat and stat.get("avg_score") is not None:
        print(f"\n🎯 avg_score: {stat['avg_score']:.4f} ({stat['avg_score']*100:.2f}%)")

    return {
        'coco_results': coco_results,
        'detailed_results': detailed_results,
        'csv_path': csv_path,
        'summary_path': summary_path,
        'bd_diag_summary': bd_diag_summary
    }

if __name__ == "__main__":
    CONFIG = {
        'model_path': r"E:\zgxf\MaizeBKN\final_experiment\train_outputs\MaizeBKN\best_model.pth",
        'test_json_path': r"E:\maize_dataset\images\annotations\test_keypoints.json",
        'images_dir': r"E:\maize_dataset\images\test",
        'output_folder': r"E:\zgxf\MaizeBKN\final_experiment\eval_results\MaizeBKN",
        'device': 'cuda:0',
        'fixed_size': [416, 624],
        'batch_size': 16,
        'num_joints': 6,
        'flip_test': False,
        'save_visualizations': True,
        'save_bd_maps': True,
        'bd_visual_limit': 50,
        'bd_visual_seed': 42,
        'eval_bd_supervision': True
    }

    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    if not os.path.exists(CONFIG['model_path']): raise FileNotFoundError(CONFIG['model_path'])
    if not os.path.exists(CONFIG['test_json_path']): raise FileNotFoundError(CONFIG['test_json_path'])
    if not os.path.exists(CONFIG['images_dir']): raise FileNotFoundError(CONFIG['images_dir'])

    res = unified_evaluation(**CONFIG)
    if res and res.get('bd_diag_summary'):
        print("\n🧪 BD Diagnostic mean：")
        for k, v in res['bd_diag_summary'].items():
            print(f"{k:<35}: {v:.4f}")
