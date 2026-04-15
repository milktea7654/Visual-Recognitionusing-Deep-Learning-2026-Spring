"""Training script for DETR digit detection."""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import config
from dataset import (
    DigitDetectionDataset,
    make_train_transforms,
    make_val_transforms,
    collate_fn,
)
from model import DETR


# ---------------------------------------------------------------------------
# Box utility functions
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU; returns (iou, union) each of shape [N, M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Generalised IoU [N, M] (both inputs in xyxy format)."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-6)
    return giou


def complete_box_iou_diag(boxes1, boxes2):
    """Complete IoU (CIoU) for matched pairs — returns [N] (diagonal only).

    Both inputs in xyxy format, same length N.
    CIoU = IoU - (d²/c²) - α·v
    where d = center distance, c = enclosing diagonal, v = aspect ratio consistency.
    """
    # IoU
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # Center distance squared
    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal squared
    enc_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enc_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    c2 = ((enc_rb[:, 0] - enc_lt[:, 0]) ** 2 +
          (enc_rb[:, 1] - enc_lt[:, 1]) ** 2).clamp(min=1e-6)

    # Aspect ratio penalty
    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - d2 / c2 - alpha * v
    return ciou


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Bipartite matching with focal-loss-based class cost (Deformable DETR)."""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Returns:
            List of (row_indices, col_indices) tuples, one per batch element.
        """
        B, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten batch dimension for cost computation
        # Sigmoid classification (no background class)
        pred_logits = outputs["pred_logits"].flatten(0, 1).sigmoid()
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([t["labels"] for t in targets])       # 1-indexed
        tgt_boxes = torch.cat([t["boxes"] for t in targets])      # cxcywh normalised

        sizes = [len(t["labels"]) for t in targets]

        if sum(sizes) == 0:
            return [(torch.tensor([], dtype=torch.long),
                     torch.tensor([], dtype=torch.long))] * B

        # --- focal-loss-based class cost (sigmoid) ---
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        neg_cost = (1 - alpha) * (pred_logits ** gamma) * (
            -(1 - pred_logits + 1e-8).log()
        )
        pos_cost = alpha * ((1 - pred_logits) ** gamma) * (
            -(pred_logits + 1e-8).log()
        )
        # tgt_ids 1-indexed → 0-indexed column selection
        cost_class = pos_cost[:, tgt_ids - 1] - neg_cost[:, tgt_ids - 1]

        # --- L1 bbox cost ---
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # --- GIoU cost ---
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )

        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        ).reshape(B, num_queries, -1)

        indices = []
        for i, (c_i, s) in enumerate(zip(C.split(sizes, -1), sizes)):
            if s == 0:
                indices.append(
                    (torch.tensor([], dtype=torch.long),
                     torch.tensor([], dtype=torch.long))
                )
            else:
                row, col = linear_sum_assignment(c_i[i].cpu().numpy())
                indices.append(
                    (torch.tensor(row, dtype=torch.long),
                     torch.tensor(col, dtype=torch.long))
                )
        return indices


# ---------------------------------------------------------------------------
# DETR loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    """
    Sigmoid focal loss for classification (Deformable DETR).

    Args:
        inputs:  [N, C] raw logits
        targets: [N, C] one-hot
        num_boxes: normaliser
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class DETRLoss(nn.Module):
    """Sigmoid focal loss + L1 + GIoU with bipartite matching + auxiliary losses."""

    def __init__(
        self,
        num_classes,
        matcher,
        loss_cls=2.0,
        loss_bbox=5.0,
        loss_giou=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls_w = loss_cls
        self.loss_bbox_w = loss_bbox
        self.loss_giou_w = loss_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _loss_single(self, outputs, targets, indices, num_boxes):
        """Compute cls + bbox + giou loss for a single set of outputs."""
        pred_logits = outputs["pred_logits"]
        B, Q, C = pred_logits.shape

        target_onehot = torch.zeros(
            (B, Q, C), dtype=torch.float32, device=pred_logits.device
        )
        for i, (row, col) in enumerate(indices):
            if len(row) > 0:
                cls_idx = targets[i]["labels"][col] - 1
                target_onehot[i, row, cls_idx] = 1.0

        loss_cls = sigmoid_focal_loss(
            pred_logits.flatten(0, 1),
            target_onehot.flatten(0, 1),
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        pred_boxes = outputs["pred_boxes"]
        loss_bbox = torch.tensor(0.0, device=pred_logits.device)
        loss_giou = torch.tensor(0.0, device=pred_logits.device)

        for i, (row, col) in enumerate(indices):
            if len(row) == 0:
                continue
            src_boxes = pred_boxes[i, row]
            tgt_boxes = targets[i]["boxes"][col]
            loss_bbox += F.l1_loss(src_boxes, tgt_boxes, reduction="sum")
            loss_giou += (
                1 - complete_box_iou_diag(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(tgt_boxes),
                )
            ).sum()

        loss_bbox /= num_boxes
        loss_giou /= num_boxes

        total = (
            self.loss_cls_w * loss_cls
            + self.loss_bbox_w * loss_bbox
            + self.loss_giou_w * loss_giou
        )
        return total, loss_cls, loss_bbox, loss_giou

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        total, loss_cls, loss_bbox, loss_giou = self._loss_single(
            outputs, targets, indices, num_boxes
        )

        # ---- auxiliary losses from intermediate decoder layers --------
        if "aux_outputs" in outputs:
            for aux_out in outputs["aux_outputs"]:
                aux_indices = self.matcher(aux_out, targets)
                aux_total, _, _, _ = self._loss_single(
                    aux_out, targets, aux_indices, num_boxes
                )
                total = total + aux_total

        # ---- class accuracy on matched pairs (final layer only) ------
        pred_logits = outputs["pred_logits"]
        num_matched = 0
        num_correct = 0
        with torch.no_grad():
            for i, (row, col) in enumerate(indices):
                if len(row) == 0:
                    continue
                pred_cls = pred_logits[i, row].argmax(-1)
                tgt_cls = targets[i]["labels"][col] - 1
                num_correct += (pred_cls == tgt_cls).sum().item()
                num_matched += len(row)

        cls_acc = num_correct / max(num_matched, 1)

        loss_dict = {
            "loss_cls": loss_cls.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
            "total": total.item(),
            "cls_acc": cls_acc,
        }
        return total, loss_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ---- Datasets & loaders -------------------------------------------
    cj_kwargs = {
        "brightness": config.COLOR_JITTER_BRIGHTNESS,
        "contrast":   config.COLOR_JITTER_CONTRAST,
        "saturation": config.COLOR_JITTER_SATURATION,
        "hue":        config.COLOR_JITTER_HUE,
    }
    train_dataset = DigitDetectionDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_JSON,
        transforms=make_train_transforms(
            config.TRAIN_SIZES,
            config.MAX_SIZE,
            use_color_jitter=config.USE_COLOR_JITTER,
            color_jitter_kwargs=cj_kwargs,
            use_random_erasing=config.USE_RANDOM_ERASING,
            random_erasing_p=config.RANDOM_ERASING_P,
            use_random_crop=config.USE_RANDOM_CROP,
            random_crop_min=config.RANDOM_CROP_MIN,
            random_crop_max=config.RANDOM_CROP_MAX,
            use_gaussian_blur=config.USE_GAUSSIAN_BLUR,
            gaussian_blur_p=config.GAUSSIAN_BLUR_P,
        ),
    )
    val_dataset = DigitDetectionDataset(
        config.VAL_IMG_DIR,
        config.VAL_JSON,
        transforms=make_val_transforms(config.VAL_SIZE, config.MAX_SIZE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

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
        pretrained_backbone=config.PRETRAINED_BACKBONE,
        use_checkpoint=config.USE_CHECKPOINT,
        aux_loss=config.AUX_LOSS,
        iterative_refine=config.ITERATIVE_REFINE,
    ).to(device)

    # ---- Optional: freeze early backbone layers -----------------------
    if config.FREEZE_BACKBONE_EARLY:
        for name, p in model.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                p.requires_grad_(False)

    # ---- Optimizer: lower LR for pretrained backbone -------------------
    backbone_params = [
        p for n, p in model.named_parameters()
        if "backbone" in n and p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]
    param_groups = [
        {"params": backbone_params, "lr": config.LR_BACKBONE},
        {"params": other_params,    "lr": config.LR},
    ]

    opt_name = config.OPTIMIZER.lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(
            param_groups, weight_decay=config.WEIGHT_DECAY
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            param_groups, weight_decay=config.WEIGHT_DECAY
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

    sched_name = config.LR_SCHEDULER.lower()
    if sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_DROP, gamma=config.LR_GAMMA
        )
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS
        )
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.LR_SCHEDULER}")

    # ---- Mixed precision scaler ----------------------------------------
    use_amp = config.AMP and torch.cuda.is_available()
    amp_dtype_str = getattr(config, 'AMP_DTYPE', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    # GradScaler is unnecessary with bf16 (no under/overflow risk)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ---- Resume from checkpoint ----------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    if config.RESUME:
        ckpt = torch.load(config.RESUME, map_location=device)
        model.load_state_dict(ckpt["model"])
        #optimizer.load_state_dict(ckpt["optimizer"])
        #scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    # ---- Loss ---------------------------------------------------------
    matcher = HungarianMatcher(
        config.COST_CLASS, config.COST_BBOX, config.COST_GIOU,
        focal_alpha=config.FOCAL_ALPHA, focal_gamma=config.FOCAL_GAMMA,
    )
    criterion = DETRLoss(
        config.NUM_CLASSES,
        matcher,
        loss_cls=config.LOSS_CLS,
        loss_bbox=config.LOSS_BBOX,
        loss_giou=config.LOSS_GIOU,
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
    ).to(device)

    writer = SummaryWriter(config.LOG_DIR)

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # ----------------------------------------------------------------
        # Train
        # ----------------------------------------------------------------
        model.train()
        train_totals = {"total": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0,
                        "loss_giou": 0.0, "cls_acc": 0.0}
        t0 = time.time()
        accum_steps = getattr(config, 'GRAD_ACCUM_STEPS', 1)

        optimizer.zero_grad()
        for step, (images, masks, targets) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False
        )):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(images, masks)
                loss, loss_dict = criterion(outputs, targets)
                loss = loss / accum_steps  # scale loss by accumulation steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            for k in train_totals:
                train_totals[k] += loss_dict[k]

        scheduler.step()

        n_train = len(train_loader)
        avg_train = train_totals["total"] / n_train
        train_metrics = {k: v / n_train for k, v in train_totals.items()}

        # ----------------------------------------------------------------
        # Validate
        # ----------------------------------------------------------------
        model.eval()
        val_totals = {"total": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0,
                      "loss_giou": 0.0, "cls_acc": 0.0}

        with torch.no_grad():
            for images, masks, targets in tqdm(
                val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False
            ):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(images, masks)
                    loss, loss_dict = criterion(outputs, targets)
                for k in val_totals:
                    val_totals[k] += loss_dict[k]

        n_val = len(val_loader)
        avg_val = val_totals["total"] / n_val
        val_metrics = {k: v / n_val for k, v in val_totals.items()}
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
            f"train_acc={train_metrics['cls_acc']:.3f}  "
            f"val_acc={val_metrics['cls_acc']:.3f}  "
            f"({elapsed:.0f}s)  "
            f"cls={val_metrics['loss_cls']:.4f}  "
            f"bbox={val_metrics['loss_bbox']:.4f}  "
            f"giou={val_metrics['loss_giou']:.4f}"
        )

        # ---- TensorBoard -----------------------------------------------
        writer.add_scalars("loss/total",
                           {"train": avg_train, "val": avg_val}, epoch)
        writer.add_scalars("loss/cls",
                           {"train": train_metrics["loss_cls"],
                            "val": val_metrics["loss_cls"]}, epoch)
        writer.add_scalars("loss/bbox",
                           {"train": train_metrics["loss_bbox"],
                            "val": val_metrics["loss_bbox"]}, epoch)
        writer.add_scalars("loss/giou",
                           {"train": train_metrics["loss_giou"],
                            "val": val_metrics["loss_giou"]}, epoch)
        writer.add_scalars("metrics/cls_acc",
                           {"train": train_metrics["cls_acc"],
                            "val": val_metrics["cls_acc"]}, epoch)
        writer.add_scalar("lr/backbone",
                          optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lr/other",
                          optimizer.param_groups[1]["lr"], epoch)

        # ----------------------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------------------
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": avg_val,
        }
        torch.save(
            ckpt,
            os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"),
        )

        cur_val_acc = val_metrics["cls_acc"]
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_val_loss = avg_val
            ckpt["val_acc"] = best_val_acc
            torch.save(ckpt, os.path.join(config.OUTPUT_DIR, "best_model.pth"))
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f}, val_loss={avg_val:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
