"""Inference script: generates pred.json from a trained DETR checkpoint."""
import os
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

import config
from model import DETR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Test dataset (no annotations)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    """Loads test images, resizes, and keeps track of original dimensions."""

    def __init__(self, img_dir, size=800, max_size=1333):
        self.img_dir = img_dir
        self.size = size
        self.max_size = max_size
        self.img_files = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith(".png")
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_id = int(os.path.splitext(fname)[0])

        image = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = image.size

        image = self._resize(image)
        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        return image, img_id, orig_h, orig_w

    def _resize(self, image):
        w, h = image.size
        min_dim, max_dim = min(h, w), max(h, w)

        scale = self.size / min_dim
        if max_dim * scale > self.max_size:
            scale = self.max_size / max_dim

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return TF.resize(image, (new_h, new_w))


def test_collate_fn(batch):
    images, img_ids, orig_hs, orig_ws = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded[i, :, :h, :w] = img
        masks[i, :h, :w] = False   # True = padding

    return padded, masks, list(img_ids), list(orig_hs), list(orig_ws)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(checkpoint_path, output_path="pred.json", score_threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model --------------------------------------------------------
    model = DETR(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        nheads=config.NHEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        n_levels=config.N_FEATURE_LEVELS,
        n_points=config.N_POINTS,
        pretrained_backbone=False,   # weights loaded from checkpoint
        aux_loss=False,              # no aux needed at inference
        iterative_refine=config.ITERATIVE_REFINE,
        two_stage=getattr(config, 'TWO_STAGE', False),
        two_stage_num_proposals=getattr(config, 'TWO_STAGE_NUM_PROPOSALS', 300),
        use_dn=getattr(config, 'USE_DN', False),
        embed_init_tgt=getattr(config, 'EMBED_INIT_TGT', False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # ---- Data ---------------------------------------------------------
    dataset = TestDataset(config.TEST_IMG_DIR, size=config.VAL_SIZE,
                          max_size=config.MAX_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=test_collate_fn,
    )

    results = []

    with torch.no_grad():
        for images, masks, img_ids, orig_hs, orig_ws in tqdm(loader, desc="Inference"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, masks)

            pred_logits = outputs["pred_logits"]   # [B, Q, C] (sigmoid cls)
            pred_boxes = outputs["pred_boxes"]     # [B, Q, 4] cxcywh normalised

            # Sigmoid activation for per-class probabilities
            probs = pred_logits.sigmoid()             # [B, Q, C]
            scores, labels = probs.max(-1)            # [B, Q]

            for i in range(len(img_ids)):
                # Find actual (non-padded) image dimensions within the batch
                not_padded = ~masks[i]
                actual_h = not_padded.any(dim=1).sum().item()
                actual_w = not_padded.any(dim=0).sum().item()

                # Model predicts in padded-tensor-normalised space
                max_h_t, max_w_t = masks.shape[1], masks.shape[2]
                scale_feat = torch.tensor(
                    [max_w_t, max_h_t, max_w_t, max_h_t],
                    dtype=torch.float32, device=device,
                )
                boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i]) * scale_feat

                # Scale from padded-tensor pixel space → original image space
                scale_orig = torch.tensor(
                    [orig_ws[i] / actual_w, orig_hs[i] / actual_h,
                     orig_ws[i] / actual_w, orig_hs[i] / actual_h],
                    dtype=torch.float32, device=device,
                )
                boxes_xyxy = boxes_xyxy * scale_orig

                # xyxy → xywh (COCO submission format)
                boxes_xywh = torch.stack([
                    boxes_xyxy[:, 0],
                    boxes_xyxy[:, 1],
                    boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                    boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
                ], dim=-1)

                keep = scores[i] > score_threshold
                for j in keep.nonzero(as_tuple=True)[0]:
                    results.append({
                        "image_id": img_ids[i],
                        "bbox": [round(v, 4) for v in boxes_xywh[j].cpu().tolist()],
                        "score": round(float(scores[i, j].item()), 6),  
                        "category_id": int(labels[i, j].item()) + 1,
                    })

    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} predictions → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR inference for digit detection")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.OUTPUT_DIR, "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pred.json",
        help="Output file path (must be named pred.json for submission)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.SCORE_THRESHOLD,
        help="Score threshold for keeping detections",
    )
    args = parser.parse_args()

    run_inference(args.checkpoint, args.output, args.threshold)
