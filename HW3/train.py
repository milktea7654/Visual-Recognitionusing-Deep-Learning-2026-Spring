"""
train.py – Training loop for Mask2Former cell instance segmentation.

Usage
-----
    python train.py                            # defaults from config.py
    python train.py --epochs 100 --batch_size 2

Loss design
-----------
Mask2Former's built-in loss is a weighted sum of:

1. **Dice Loss** (per-query binary mask vs. GT binary mask)
   - Smooth, gradient-friendly for sparse masks; handles class imbalance
     because it is normalised by the sum of predictions + targets.

2. **Cross-Entropy / Focal Loss** (per-pixel mask logit)
   - ``sigmoid_focal_loss`` down-weights easy background pixels; critical when
     cells occupy <5 % of image area.

3. **Classification Cross-Entropy** (per-query class prediction)
   - ``no_object`` weight (0.1) down-weights the ~90 dead queries per image.

All three are combined inside ``Mask2FormerForUniversalSegmentation`` via
bipartite Hungarian matching – no NMS needed.

VRAM reduction strategy
-----------------------
* ``USE_AMP = True``   – fp16 autocast halves activation memory (~50 %).
* ``IMAGE_SIZE = 384`` – reduces spatial tensors by (384/512)^2 ≈ 44 %.
* ``MAX_INSTANCES``    – caps mask_labels tensor rows per image.

Optimiser & scheduler
---------------------
* **AdamW** with decoupled weight decay (0.05): standard for ViT/Swin models.
* **OneCycleLR** (pct_start=0.2): 60-epoch linear warmup then cosine anneal.
  Proven to converge faster on small datasets vs flat LR schedules.
* Backbone is frozen for the first ``WARMUP_EPOCHS`` epochs then unfrozen.
"""

from __future__ import annotations

import argparse
import datetime
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import CellInstanceDataset, collate_fn
from model import Mask2FormerWrapper


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Smoothed metric tracking
# ---------------------------------------------------------------------------


class _Meter:
    """Running-window average for a single scalar."""

    def __init__(self, win: int = 20) -> None:
        self._d: deque = deque(maxlen=win)
        self._sum = 0.0
        self._n = 0

    def update(self, v: float, n: int = 1) -> None:
        self._d.append(v)
        self._sum += v * n
        self._n += n

    @property
    def latest(self) -> float:
        return float(self._d[-1]) if self._d else 0.0

    @property
    def median(self) -> float:
        return float(np.median(list(self._d))) if self._d else 0.0

    @property
    def global_avg(self) -> float:
        return self._sum / max(self._n, 1)


# ---------------------------------------------------------------------------
# Per-image target preparation
# ---------------------------------------------------------------------------


def _prepare_targets(
    targets: List[dict],
    device: torch.device,
) -> tuple:
    """
    Convert a list of per-image target dicts to two lists accepted by HF
    Mask2Former:
        mask_labels:  list[(N_i, H, W) float32]
        class_labels: list[(N_i,) int64]  – 0-indexed (class - 1)

    Caps N_i at config.MAX_INSTANCES (random subset) to prevent OOM on
    densely annotated images.
    """
    mask_labels = []
    class_labels = []

    for tgt in targets:
        masks = tgt["masks"].float().to(device)   # (N, H, W)
        labels = (tgt["labels"] - 1).long().to(device)  # (N,)

        if masks.shape[0] > config.MAX_INSTANCES:
            idx = torch.randperm(masks.shape[0], device=device)[:config.MAX_INSTANCES]
            masks = masks[idx]
            labels = labels[idx]

        mask_labels.append(masks)
        class_labels.append(labels)

    return mask_labels, class_labels


# ---------------------------------------------------------------------------
# Train / val epoch runners
# ---------------------------------------------------------------------------


def _run_train_epoch(
    model: Mask2FormerWrapper,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter,
    global_step: int,
    use_amp: bool,
) -> tuple:
    """Run one training epoch. Returns (avg_loss, updated_global_step)."""
    model.train()
    meter = _Meter(win=20)

    pbar = tqdm(loader, desc=f"Train {epoch:03d}/{total_epochs:03d}", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"]

        if not targets:
            continue

        mask_labels, class_labels = _prepare_targets(targets, device)
        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            outputs = model(
                pixel_values=images,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        meter.update(loss_val)
        global_step += 1

        if writer is not None and global_step % 10 == 0:
            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[1]["lr"], global_step)

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            avg=f"{meter.global_avg:.4f}",
            lr=f"{optimizer.param_groups[1]['lr']:.2e}",
        )

    return meter.global_avg, global_step


@torch.no_grad()
def _run_val_epoch(
    model: Mask2FormerWrapper,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float = 0.5,
) -> float:
    """Quick mask-IoU AP50 estimate over the validation set."""
    model.eval()
    all_iou: List[float] = []

    pbar = tqdm(loader, desc="Val      ", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"]

        with autocast('cuda', enabled=use_amp):
            outputs = model(pixel_values=images)

        pred_masks_logits = outputs.masks_queries_logits  # (B, Q, H/4, W/4)
        pred_class_logits = outputs.class_queries_logits  # (B, Q, C+1)

        h, w = images.shape[2], images.shape[3]
        pred_masks = torch.nn.functional.interpolate(
            pred_masks_logits.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()  # (B, Q, H, W)

        pred_scores = pred_class_logits.float().softmax(-1)
        fg_scores, fg_labels = pred_scores[..., :-1].max(-1)  # (B, Q)

        for b_idx, tgt in enumerate(targets):
            gt_masks = tgt["masks"].to(device, non_blocking=True)
            if gt_masks.shape[0] == 0:
                continue

            img_pred_masks = pred_masks[b_idx]
            img_scores = fg_scores[b_idx]

            keep = img_scores > threshold
            if keep.sum() == 0:
                all_iou.extend([0.0] * gt_masks.shape[0])
                continue

            pos_masks = (img_pred_masks[keep] > 0.5)

            for gt_m in gt_masks:
                gt_m = gt_m.bool()
                inter = (pos_masks & gt_m.unsqueeze(0)).float().sum((-2, -1))
                union = (pos_masks | gt_m.unsqueeze(0)).float().sum((-2, -1))
                iou = (inter / (union + 1e-6)).max().item()
                all_iou.append(float(iou >= 0.5))

        cur_ap50 = float(np.mean(all_iou)) if all_iou else 0.0
        pbar.set_postfix(ap50=f"{cur_ap50:.4f}")

    return float(np.mean(all_iou)) if all_iou else 0.0


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    _seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train] device: {device}")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_dir = Path(config.DATA_ROOT)
    all_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    n_total = len(all_dirs)
    n_val = max(1, int(n_total * 0.2))
    rng = torch.Generator().manual_seed(config.SEED)
    perm = torch.randperm(n_total, generator=rng).tolist()
    train_sample_dirs = [all_dirs[i] for i in perm[n_val:]]
    val_sample_dirs = [all_dirs[i] for i in perm[:n_val]]

    train_ds = CellInstanceDataset(
        str(train_dir),
        augment=True,
        crop_size=config.CROP_SIZE,
        min_size=config.MIN_SIZE,
        size_divisibility=config.SIZE_DIVISIBILITY,
    )
    train_ds.sample_dirs = train_sample_dirs

    val_ds = CellInstanceDataset(
        str(train_dir),
        augment=False,
        crop_size=config.CROP_SIZE,
        min_size=config.MIN_SIZE,
        size_divisibility=config.SIZE_DIVISIBILITY,
    )
    val_ds.sample_dirs = val_sample_dirs

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = Mask2FormerWrapper(
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
    ).to(device)

    if config.WARMUP_EPOCHS > 0:
        model.freeze_backbone()
        print(f"[train] backbone frozen for first {config.WARMUP_EPOCHS} epochs")

    # ── Optimiser ────────────────────────────────────────────────────────────
    backbone_params = [
        p for n, p in model.named_parameters()
        if "pixel_level_module.encoder" in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "pixel_level_module.encoder" not in n
    ]

    param_groups = [
        {"params": backbone_params, "lr": config.LR_BACKBONE},
        {"params": head_params,     "lr": config.LR},
    ]

    if getattr(config, "OPTIMIZER", "AdamW") == "AdamW":
        optimizer = AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    else:
        # Fallback to Adam if requested
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)

    if getattr(config, "SCHEDULER", "OneCycleLR") == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config.LR_BACKBONE, config.LR],
            epochs=args.epochs,
            steps_per_epoch=1,
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=config.LR_MIN,
        )

    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = GradScaler('cuda', enabled=use_amp)
    print(f"[train] AMP={'on' if use_amp else 'off'}")

    # ── Output dirs ──────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.OUTPUT_DIR) / stamp
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[train] run dir    → {run_dir}")
    print(f"[train] TensorBoard → tensorboard --logdir {log_dir}")

    best_ap50 = 0.0
    best_ckpt = ckpt_dir / "best_model.pth"
    global_step = 0
    no_improve = 0
    patience = getattr(config, "EARLY_STOP_PATIENCE", 20)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if epoch == config.WARMUP_EPOCHS + 1:
            model.unfreeze_backbone()
            print(f"[train] epoch {epoch}: backbone unfrozen")

        ep_t0 = time.time()

        avg_loss, global_step = _run_train_epoch(
            model, train_loader, optimizer, scaler,
            device, epoch, args.epochs, writer, global_step, use_amp,
        )
        scheduler.step()

        ap50 = _run_val_epoch(
            model, val_loader, device, use_amp,
            threshold=config.VAL_SCORE_THRESHOLD
        )
        ep_sec = int(time.time() - ep_t0)
        lr_now = optimizer.param_groups[1]["lr"]

        print(
            f"  Epoch {epoch:03d}  train_loss={avg_loss:.4f}  "
            f"AP50={ap50:.4f}  lr={lr_now:.2e}  ({ep_sec}s)"
        )

        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("val/AP50", ap50, epoch)
            writer.add_scalar("train/lr_epoch", lr_now, epoch)

        # Periodic checkpoint
        if epoch % config.SAVE_EVERY == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ap50": ap50,
                },
                ckpt_path,
            )

        # Save best model / early stopping
        if ap50 > best_ap50:
            best_ap50 = ap50
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ap50": ap50,
                },
                best_ckpt,
            )
            print(f"  -> Best model saved (AP50={best_ap50:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"\n[train] early stopping at epoch {epoch} "
                    f"(no AP50 improvement for {patience} epochs)"
                )
                break

    writer.close()
    print(f"\n[train] finished – best AP50: {best_ap50:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mask2Former for cell instance segmentation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="total training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="images per GPU batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.NUM_WORKERS,
        help="DataLoader workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
