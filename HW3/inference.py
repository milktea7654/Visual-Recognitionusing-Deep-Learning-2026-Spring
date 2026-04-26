"""
inference.py – Mask R-CNN inference with sliding window (SAHI), TTA, and Soft-NMS.

Output format (test-results.json, COCO-style detection):
    [{
        "image_id": int,
        "bbox": [x, y, w, h],
        "score": float,
        "category_id": int,
        "segmentation": {"size": [H, W], "counts": "..."},   # RLE
    }, ...]

Usage:
    python inference.py --checkpoint output/<run>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from torchvision.ops import nms
from tqdm import tqdm

import config
from model import build_maskrcnn, build_cascade_maskrcnn

try:
    from ensemble_boxes import weighted_boxes_fusion
    _HAS_WBF = True
except ImportError:
    _HAS_WBF = False
    print("WARNING: ensemble_boxes not installed. Install with: pip install ensemble-boxes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask R-CNN inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_dir", type=str, default=config.TEST_DIR)
    p.add_argument("--id_map",   type=str, default=config.TEST_ID_MAP)
    p.add_argument("--output",   type=str, default="test-results.json")
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--score_thr", type=float, default=0.05)
    p.add_argument("--no_tta",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def _tile_coords(H: int, W: int, tile: int, overlap: float) -> List[Tuple[int, int]]:
    if H <= tile and W <= tile:
        return [(0, 0)]
    step = max(1, int(tile * (1.0 - overlap)))
    ys = list(range(0, max(1, H - tile + 1), step))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), step))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    return [(y, x) for y in ys for x in xs]


# ---------------------------------------------------------------------------
# Tile predictions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tile predictions
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict_tile_single(model, tile_np: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Predict on a single tile variant (no TTA).
    Returns (boxes, scores, labels, masks_np) all in tile coordinate frame.
    masks_np is uint8 numpy (N, H, W) – binarized on GPU then transferred to CPU.
    """
    t = torch.from_numpy(tile_np).permute(2, 0, 1).float().div_(255.0).to(device)
    out = model([t])[0]
    boxes  = out["boxes"].cpu()
    scores = out["scores"].cpu()
    labels = out["labels"].cpu()
    # Binarize on GPU then move to CPU
    masks_np = (out["masks"].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return boxes, scores, labels, masks_np


@torch.no_grad()
def _predict_tile_tta(model, tile_np: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Predict on tile with TTA (orig, hflip, vflip).
    Returns concatenated predictions from all variants (NMS will deduplicate later).
    
    Returns (boxes, scores, labels, masks_np) all in tile coordinate frame.
    """
    H, W = tile_np.shape[:2]
    
    # Collect predictions from all 3 TTA variants
    all_b, all_s, all_l, all_m = [], [], [], []
    
    variants = [
        ("orig", tile_np),
        ("hflip", tile_np[:, ::-1, :].copy()),
        ("vflip", tile_np[::-1, :, :].copy()),
    ]
    
    for variant_name, arr in variants:
        boxes, scores, labels, masks_np = _predict_tile_single(model, arr, device)
        
        # Flip back to original coords
        if variant_name == "hflip":
            masks_np = masks_np[:, :, ::-1].copy()
            x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0] = W - x2
            boxes[:, 2] = W - x1
        elif variant_name == "vflip":
            masks_np = masks_np[:, ::-1, :].copy()
            y1, y2 = boxes[:, 1].clone(), boxes[:, 3].clone()
            boxes[:, 1] = H - y2
            boxes[:, 3] = H - y1
        
        all_b.append(boxes)
        all_s.append(scores)
        all_l.append(labels)
        all_m.append(masks_np)
    
    if not all_b:
        return (
            torch.zeros((0, 4)), torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.int64), np.zeros((0, H, W), np.uint8),
        )
    return (
        torch.cat(all_b, dim=0),
        torch.cat(all_s, dim=0),
        torch.cat(all_l, dim=0),
        np.concatenate(all_m, axis=0),
    )


@torch.no_grad()
def predict_image(
    model, img_np: np.ndarray, device,
    tile: int, overlap: float, tta: bool, score_thr: float,
):

    H, W = img_np.shape[:2]

    # ── Step 1: collect per-detection data ──────────────────────────────────
    all_boxes:  list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    det_placement: list[tuple] = []

    def _register(tb, ts, tl, tm_np, off_y, off_x, th, tw):
        """Shift boxes to global coords and record tile-local masks."""
        tb = tb.clone()
        tb[:, [0, 2]] += off_x
        tb[:, [1, 3]] += off_y
        tb[:, 0].clamp_(0, W)
        tb[:, 2].clamp_(0, W)
        tb[:, 1].clamp_(0, H)
        tb[:, 3].clamp_(0, H)
        all_boxes.append(tb)
        all_scores.append(ts)
        all_labels.append(tl)
        for i in range(tm_np.shape[0]):
            det_placement.append((off_y, off_x, th, tw, tm_np[i]))

    # Choose prediction function based on TTA setting
    predict_fn = _predict_tile_tta if tta else _predict_tile_single

    if H <= tile and W <= tile:
        # Pad to tile×tile to match training distribution.
        if H < tile or W < tile:
            padded = np.zeros((tile, tile, 3), dtype=img_np.dtype)
            padded[:H, :W] = img_np
            inp = padded
        else:
            inp = img_np
        tb, ts, tl, tm = predict_fn(model, inp, device) if tta else _predict_tile_single(model, inp, device)
        if tb.shape[0] > 0:
            _register(tb, ts, tl, tm[:, :H, :W], 0, 0, H, W)
    else:
        for (y, x) in _tile_coords(H, W, tile, overlap):
            tile_np = img_np[y:y + tile, x:x + tile]
            th, tw = tile_np.shape[:2]
            if th < tile or tw < tile:
                pad = np.zeros((tile, tile, 3), dtype=tile_np.dtype)
                pad[:th, :tw] = tile_np
                tile_np = pad
            tb, ts, tl, tm = predict_fn(model, tile_np, device) if tta else _predict_tile_single(model, tile_np, device)
            if tb.shape[0] == 0:
                continue
            _register(tb, ts, tl, tm[:, :th, :tw], y, x, th, tw)

    if not all_boxes:
        return [], [], [], np.zeros((0, H, W), np.uint8)

    full_boxes = torch.cat(all_boxes,  dim=0)
    scores     = torch.cat(all_scores, dim=0)
    labels     = torch.cat(all_labels, dim=0)

    # ── Step 2: score filter ─────────────────────────────────────────────────
    score_keep = (scores >= score_thr).numpy()
    full_boxes = full_boxes[score_keep]
    scores     = scores[score_keep]
    labels     = labels[score_keep]
    det_placement = [det_placement[i] for i in range(len(det_placement)) if score_keep[i]]

    if full_boxes.shape[0] == 0:
        return [], [], [], np.zeros((0, H, W), np.uint8)

    # ── Step 3: per-class hard NMS ──────────────────────────────────────────
    # WBF already handles fusion; just apply hard NMS (IoU=0.5) as final cleanup.
    final_idx_all = []
    for c in torch.unique(labels):
        m = labels == c
        local_idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        keep_hard = nms(full_boxes[m].float(), scores[m].float(), iou_threshold=0.5)
        final_idx_all.append(local_idx[keep_hard])

    if not final_idx_all:
        return [], [], [], np.zeros((0, H, W), np.uint8)
    final_idx = torch.cat(final_idx_all, dim=0).numpy()

    # ── Step 4: assemble full-image masks ───────────────────────────────────
    final_masks = []
    for idx in final_idx:
        oy, ox, th, tw, small = det_placement[idx]
        full_m = np.zeros((H, W), np.uint8)
        full_m[oy:oy + th, ox:ox + tw] = small[:th, :tw]
        final_masks.append(full_m)

    return (
        full_boxes[final_idx].tolist(),
        scores[final_idx].tolist(),
        labels[final_idx].tolist(),
        np.stack(final_masks, axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    use_tta = config.USE_TTA and not args.no_tta
    print(f"Using TTA={use_tta}, WBF={config.USE_WBF}, score_thr={args.score_thr}")

    # ----- Model -----
    if getattr(config, "USE_CASCADE", False):
        model = build_cascade_maskrcnn(pretrained_backbone=False, multi_scale=False).to(device)
    else:
        model = build_maskrcnn(pretrained_backbone=False, multi_scale=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Prefer EMA weights if present
    state_dict = None
    if isinstance(ckpt, dict):
        if ckpt.get("ema") is not None:
            state_dict = ckpt["ema"]
            print("Loaded EMA weights")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print("Loaded model weights")
    if state_dict is None:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()

    # ----- ID map -----
    id_map_path = Path(args.id_map)
    if id_map_path.exists():
        with open(id_map_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            id_map = {item["file_name"]: int(item["id"]) for item in raw}
        else:
            id_map = {k: int(v) for k, v in raw.items()}
    else:
        print(f"WARNING: ID map not found at {id_map_path}; using filename stems as ids.")
        id_map = {}

    # ----- Iterate test images -----
    test_dir = Path(args.test_dir)
    img_paths = sorted(test_dir.glob("*.tif"))
    print(f"Found {len(img_paths)} test images")

    results: List[Dict] = []
    for img_path in tqdm(img_paths, desc="Inference", unit="img"):
        fname = img_path.name
        if fname in id_map:
            image_id = id_map[fname]
        elif img_path.stem in id_map:
            image_id = id_map[img_path.stem]
        else:
            try:
                image_id = int(img_path.stem)
            except ValueError:
                continue

        img = _load_image_tif(img_path)
        boxes, scores, labels, masks = predict_image(
            model, img, device,
            tile=config.CROP_SIZE,
            overlap=config.SLIDING_WINDOW_OVERLAP,
            tta=use_tta,
            score_thr=args.score_thr,
        )

        for b, sc, lbl, m in zip(boxes, scores, labels, masks):
            x1, y1, x2, y2 = b
            rle = coco_mask.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("utf-8")
            results.append({
                "image_id": int(image_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
                "category_id": int(lbl),
                "segmentation": rle,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} detections → {out_path}")


if __name__ == "__main__":
    main()
