"""
DINO inference – produces pred.json in COCO format.

Usage:
    python inference.py --checkpoint output/.../best.pth [--test_dir data/test]
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from config import get_config
from model import build_model
from dataset import AugPipeline, ScaleJitter, ConvertToTensor, NormalizeBoxes


def _val_pipeline(cfg):
    se = cfg.get("val_short_edge", 320)
    ms = cfg.get("val_max_size", 640)
    return AugPipeline(
        [ScaleJitter([se], max_side=ms),
         ConvertToTensor(),
         NormalizeBoxes()])


@torch.no_grad()
def run(ckpt_path, test_dir=None, out_path="pred.json", thr=0.0, bs=1):
    cfg = get_config()
    dev = torch.device(cfg["device"])

    model, _, post = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    if test_dir is None:
        test_dir = os.path.join(cfg["data_path"], "test")

    tfm = _val_pipeline(cfg)
    files = sorted(Path(test_dir).glob("*.png"))
    if not files:
        files = sorted(Path(test_dir).glob("*.jpg"))
    print(f"{len(files)} test images in {test_dir}")

    dets = []
    for i in tqdm(range(0, len(files), bs), desc="Inference"):
        batch_paths = files[i:i + bs]
        tensors, hw, ids = [], [], []
        for fp in batch_paths:
            ids.append(int(fp.stem))
            im = Image.open(fp).convert("RGB")
            hw.append((im.height, im.width))
            t, _ = tfm(im, None)
            tensors.append(t)

        inp = [t.to(dev) for t in tensors]
        out = model(inp)

        sizes = torch.tensor(hw, device=dev)
        results = post["bbox"](out, sizes)

        for r, img_id in zip(results, ids):
            mask = r["scores"] > thr
            for s, l, b in zip(r["scores"][mask], r["labels"][mask],
                               r["boxes"][mask]):
                x1, y1, x2, y2 = b.tolist()
                dets.append({
                    "image_id": img_id,
                    "bbox": [
                        round(x1, 4),
                        round(y1, 4),
                        round(x2 - x1, 4),
                        round(y2 - y1, 4)
                    ],
                    "score": round(s.item(), 6),
                    "category_id": l.item(),
                })

    with open(out_path, "w") as f:
        json.dump(dets, f)
    print(f"Saved {len(dets)} detections → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("DINO Inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--output", default="pred.json")
    ap.add_argument("--score_threshold", default=0.01, type=float)
    ap.add_argument("--batch_size", default=1, type=int)
    a = ap.parse_args()
    run(a.checkpoint, a.test_dir, a.output, a.score_threshold, a.batch_size)
