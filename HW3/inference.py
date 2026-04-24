"""
inference.py – Post-processing and COCO RLE submission generation.

Usage
-----
    python inference.py --checkpoint output/checkpoints/best_model.pth \\
                        --test_dir   data/test_release \\
                        --out        predictions.json

Output format
-------------
A JSON file containing a list of COCO-style detection entries::

    [
        {
            "image_id":    1,
            "category_id": 2,
            "segmentation": {"size": [H, W], "counts": "<RLE string>"},
            "score":       0.93
        },
        ...
    ]

``segmentation`` uses COCO *binary mask RLE* encoding via ``pycocotools``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask

import config
from dataset import CellTestDataset
from model import Mask2FormerWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_rle(binary_mask: np.ndarray) -> Dict:
    """
    Encode a boolean / uint8 (H, W) mask into COCO RLE format.

    Args:
        binary_mask: (H, W) np.uint8 array with values 0 or 1.

    Returns:
        dict with keys ``"size"`` ([H, W]) and ``"counts"`` (UTF-8 string).
    """
    # Fortran-order (column-major) required by pycocotools
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # counts is bytes in Python 3; decode for JSON serialisability
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _load_image_id_map(json_path: str) -> Dict[str, int]:
    """
    Load the test_image_name_to_ids.json mapping.

    Returns:
        {file_name: image_id}
    """
    with open(json_path, "r") as fh:
        entries = json.load(fh)
    return {e["file_name"]: e["id"] for e in entries}


# ---------------------------------------------------------------------------
# Sliding-window inference for large images
# ---------------------------------------------------------------------------


def _make_tiles(
    h: int,
    w: int,
    tile_size: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate (top, left, bottom, right) tile coordinates that cover the full
    image with ``overlap`` fractional overlap between adjacent tiles.
    """
    stride = int(tile_size * (1.0 - overlap))
    stride = max(stride, 1)
    tiles = []
    y = 0
    while True:
        x = 0
        while True:
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            tiles.append((y1, x1, y2, x2))
            if x2 == w:
                break
            x += stride
        if y2 == h:
            break
        y += stride
    return tiles


@torch.no_grad()
def _sliding_window_single(
    model: Mask2FormerWrapper,
    image_full: torch.Tensor,
    tile_size: int,
    overlap: float,
    size_divisibility: int,
    score_threshold: float,
    mask_threshold: float,
    device: torch.device,
) -> List[Dict]:
    """
    Run tiled inference on one large image (C, H, W) float32 (already
    ImageNet-normalised). Returns list of detection dicts (no image_id).
    """
    _, full_h, full_w = image_full.shape
    tiles = _make_tiles(full_h, full_w, tile_size, overlap)

    # Accumulate binary mask logits on CPU to save VRAM
    # score_canvas[q, h, w] would be too large; instead collect per-tile detections
    # and do mask-level NMS at the end.
    all_detections: List[Dict] = []

    d = size_divisibility
    for (y1, x1, y2, x2) in tiles:
        tile = image_full[:, y1:y2, x1:x2]           # (3, th, tw)
        _, th, tw = tile.shape

        # Pad tile to size_divisibility
        pad_h = (math.ceil(th / d) * d) - th
        pad_w = (math.ceil(tw / d) * d) - tw
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as _F
            tile_padded = _F.pad(tile, (0, pad_w, 0, pad_h), value=0.0)
        else:
            tile_padded = tile
        padded_th, padded_tw = tile_padded.shape[1], tile_padded.shape[2]

        inp = tile_padded.unsqueeze(0).to(device)     # (1, 3, pH, pW)
        outputs = model(pixel_values=inp)
        pred_masks_logits = outputs.masks_queries_logits[0]  # (Q, He, We)
        pred_class_logits = outputs.class_queries_logits[0]  # (Q, C+1)

        tile_dets = _postprocess_single(
            pred_masks_logits=pred_masks_logits,
            pred_class_logits=pred_class_logits,
            orig_h=th,
            orig_w=tw,
            padded_h=padded_th,
            padded_w=padded_tw,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

        # Embed tile masks into full-image canvas
        for det in tile_dets:
            rle = det["segmentation"]
            tile_mask = coco_mask.decode({
                "size": rle["size"],
                "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"],
            }).astype(np.uint8)  # (th, tw)

            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = tile_mask

            all_detections.append({
                "category_id": det["category_id"],
                "score": det["score"],
                "_mask_np": full_mask,
            })

    if not all_detections:
        return []

    # ── Mask-level NMS: remove duplicates from overlapping tiles ────────────
    # Sort by score descending, then greedy mask IoU suppression.
    all_detections.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = [False] * len(all_detections)

    for i, det_i in enumerate(all_detections):
        if suppressed[i]:
            continue
        kept.append(det_i)
        mask_i = det_i["_mask_np"].astype(bool)
        area_i = mask_i.sum()
        for j in range(i + 1, len(all_detections)):
            if suppressed[j]:
                continue
            mask_j = all_detections[j]["_mask_np"].astype(bool)
            inter = (mask_i & mask_j).sum()
            if inter == 0:
                continue
            union = (mask_i | mask_j).sum()
            iou = inter / (union + 1e-6)
            if iou > 0.5:
                suppressed[j] = True

    # Encode final masks to RLE
    results = []
    for det in kept:
        rle = _encode_rle(det["_mask_np"])
        results.append({
            "category_id": det["category_id"],
            "segmentation": rle,
            "score": det["score"],
        })
    return results


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _postprocess_single(
    pred_masks_logits: torch.Tensor,
    pred_class_logits: torch.Tensor,
    orig_h: int,
    orig_w: int,
    padded_h: int,
    padded_w: int,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    max_detections: int = 300,
) -> List[Dict]:
    """
    Convert raw Mask2Former output for ONE image into a list of detection
    dicts (before inserting ``image_id``).

    Args:
        pred_masks_logits: (Q, H_enc, W_enc) – raw logits from the decoder.
        pred_class_logits: (Q, C+1) – raw class logits.
        orig_h, orig_w:    original image dimensions (before padding).
        padded_h, padded_w: padded input size fed to the model.
        score_threshold:   minimum confidence to keep a prediction.
        mask_threshold:    sigmoid threshold to binarise the mask.
        max_detections:    hard cap on returned predictions per image.

    Returns:
        list of dicts: {"category_id", "segmentation", "score"}.
    """
    num_queries, h_enc, w_enc = pred_masks_logits.shape
    num_classes_plus_bg = pred_class_logits.shape[-1]
    num_classes = num_classes_plus_bg - 1  # last index = no-object

    # Class probabilities (softmax) and foreground scores
    class_probs = pred_class_logits.softmax(-1)          # (Q, C+1)
    fg_probs = class_probs[..., :num_classes]            # (Q, C)
    fg_scores, fg_labels = fg_probs.max(-1)              # (Q,), (Q,)

    # Mask sigmoid at encoder resolution, then upsample to padded size, then
    # crop top-left back to original size (removing zero-padding)
    masks_prob = pred_masks_logits.sigmoid()             # (Q, H_enc, W_enc)
    masks_padded = F.interpolate(
        masks_prob.unsqueeze(0),                         # (1, Q, H_enc, W_enc)
        size=(padded_h, padded_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                                         # (Q, padded_H, padded_W)
    masks_full = masks_padded[:, :orig_h, :orig_w]       # (Q, orig_H, orig_W)

    # Combined score: class confidence × mask confidence (averaged over the foreground)
    mask_pred = (masks_full > mask_threshold).float()
    mask_confidence = (masks_full * mask_pred).sum(dim=(-2, -1)) / (
        mask_pred.sum(dim=(-2, -1)) + 1e-6
    )  # (Q,)
    combined_score = fg_scores * mask_confidence         # (Q,)

    # Filter
    keep = combined_score > score_threshold
    if keep.sum() == 0:
        return []

    kept_masks = masks_full[keep]              # (K, H, W)
    kept_labels = fg_labels[keep]              # (K,)  0-indexed
    kept_scores = combined_score[keep]         # (K,)

    # Sort by score descending, cap at max_detections
    order = kept_scores.argsort(descending=True)
    order = order[:max_detections]
    kept_masks = kept_masks[order]
    kept_labels = kept_labels[order]
    kept_scores = kept_scores[order]

    detections = []
    for i in range(kept_masks.shape[0]):
        binary_mask = (kept_masks[i] > mask_threshold).cpu().numpy().astype(
            np.uint8
        )
        if binary_mask.sum() == 0:
            continue
        rle = _encode_rle(binary_mask)
        detections.append(
            {
                "category_id": int(kept_labels[i].item()) + 1,  # 1-indexed
                "segmentation": rle,
                "score": float(kept_scores[i].item()),
            }
        )

    return detections


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    checkpoint_path: str,
    test_dir: str,
    id_map_path: str,
    output_path: str,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device_str: str = "cuda",
) -> None:
    """
    Run Mask2Former inference on the test set and save COCO JSON.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint.
        test_dir:        Directory containing test ``.tif`` images.
        id_map_path:     Path to ``test_image_name_to_ids.json``.
        output_path:     Where to write ``predictions.json``.
        score_threshold: Minimum score to keep a detection.
        mask_threshold:  Sigmoid threshold for binary masks.
        device_str:      ``"cuda"`` or ``"cpu"``.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    model = Mask2FormerWrapper(
        pretrained=False,
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[inference] loaded checkpoint: {checkpoint_path}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = CellTestDataset(
        root_dir=test_dir,
        max_size=getattr(config, "MAX_SIZE", 768),
        min_size=getattr(config, "MIN_SIZE", 256),
        size_divisibility=getattr(config, "SIZE_DIVISIBILITY", 32),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    id_map = _load_image_id_map(id_map_path)
    tile_size = getattr(config, "CROP_SIZE", 768)
    sw_overlap = getattr(config, "SLIDING_WINDOW_OVERLAP", 0.2)
    size_div   = getattr(config, "SIZE_DIVISIBILITY", 32)

    # ── Inference loop ───────────────────────────────────────────────────────
    all_results: List[Dict] = []

    for batch in loader:
        image = batch["image"]               # (1, 3, H_padded, W_padded)
        file_name = batch["file_name"][0]
        orig_h = int(batch["orig_h"][0].item())
        orig_w = int(batch["orig_w"][0].item())
        padded_h = int(batch["padded_h"][0].item())
        padded_w = int(batch["padded_w"][0].item())

        image_id = id_map.get(file_name)
        if image_id is None:
            print(f"  [warn] {file_name} not found in id map – skipping")
            continue

        # Choose path: sliding window for large images, direct forward for small.
        if orig_h > tile_size or orig_w > tile_size:
            # Re-load the normalised full-resolution image without any resize.
            # CellTestDataset already stores the normalised image; since it was
            # padded (not resized) we can crop back to orig_h × orig_w safely.
            image_norm = image[0, :, :orig_h, :orig_w]  # (3, orig_H, orig_W)
            detections = _sliding_window_single(
                model=model,
                image_full=image_norm,
                tile_size=tile_size,
                overlap=sw_overlap,
                size_divisibility=size_div,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                device=device,
            )
        else:
            outputs = model(pixel_values=image.to(device))
            pred_masks_logits = outputs.masks_queries_logits[0]
            pred_class_logits = outputs.class_queries_logits[0]
            detections = _postprocess_single(
                pred_masks_logits=pred_masks_logits,
                pred_class_logits=pred_class_logits,
                orig_h=orig_h,
                orig_w=orig_w,
                padded_h=padded_h,
                padded_w=padded_w,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )

        for det in detections:
            det["image_id"] = image_id
            all_results.append(det)

        print(
            f"  {file_name}  image_id={image_id}  "
            f"detections={len(detections)}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh)

    print(f"\n[inference] saved {len(all_results)} detections → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask2Former inference + COCO RLE export"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoints/best_model.pth",
        help="path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test_release",
        help="directory of test .tif images",
    )
    parser.add_argument(
        "--id_map",
        type=str,
        default="data/test_image_name_to_ids.json",
        help="test_image_name_to_ids.json path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/predictions.json",
        help="output COCO JSON path",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.5,
        help="confidence threshold for detections",
    )
    parser.add_argument(
        "--mask_thr",
        type=float,
        default=0.5,
        help="sigmoid threshold to binarise predicted masks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        id_map_path=args.id_map,
        output_path=args.out,
        score_threshold=args.score_thr,
        mask_threshold=args.mask_thr,
    )
