"""Train a configurable PromptIR-style model for HW4 rain/snow restoration."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]
from tqdm import tqdm

from dataset import RainSnowTrainDataset, scan_train_pairs, split_pairs


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    includes = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for item in includes:
        include_path = Path(item)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        deep_update(merged, load_config(include_path))
    deep_update(merged, cfg)
    return merged


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW4 image restoration model.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data_root", type=str, default="", help="Override data.root.")
    parser.add_argument("--resume", type=str, default="", help="Override train.resume.")
    parser.add_argument("--run_name", type=str, default="", help="Override project.run_name.")
    parser.add_argument("--output_root", type=str, default="", help="Override project.output_root.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dotted keys. Example: --set model.dim=64 --set train.epochs=400",
    )
    return parser.parse_args()


def prepare_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        set_by_dotted_key(cfg, "data.root", args.data_root)
    if args.resume:
        set_by_dotted_key(cfg, "train.resume", args.resume)
    if args.run_name:
        set_by_dotted_key(cfg, "project.run_name", args.run_name)
    if args.output_root:
        set_by_dotted_key(cfg, "project.output_root", args.output_root)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_key(cfg, key, value)
    return cfg


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def tensor_psnr(pred: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def average_logs(logs: Iterable[Dict[str, float]]) -> Dict[str, float]:
    logs = list(logs)
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: sum(item[key] for item in logs) / len(logs) for key in keys}


def format_logs(logs: Dict[str, float]) -> str:
    return " | ".join(f"{key}: {value:.5f}" for key, value in logs.items())


def create_run_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(cfg.get("project", {}).get("run_name", "")).strip()
    folder_name = timestamp if not run_name else f"{timestamp}_{run_name}"
    run_dir = mkdir(Path(cfg.get("project", {}).get("output_root", "outputs")) / folder_name)
    dirs = {
        "run": run_dir,
        "checkpoints": mkdir(run_dir / "checkpoints"),
        "logs": mkdir(run_dir / "logs"),
    }
    return dirs


# -----------------------------------------------------------------------------
# Inference utilities used for validation
# -----------------------------------------------------------------------------


def pad_to_multiple(x: Tensor, multiple: int = 8) -> Tuple[Tensor, Tuple[int, int]]:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (height, width)


@torch.no_grad()
def simple_inference(model: nn.Module, x: Tensor, multiple: int = 8) -> Tensor:
    x_pad, (height, width) = pad_to_multiple(x, multiple)
    out = model(x_pad)
    return out[:, :, :height, :width]


def _build_weight(tile_h: int, tile_w: int, device: torch.device) -> Tensor:
    y = torch.hann_window(tile_h, periodic=False, device=device).clamp_min(1e-3)
    x = torch.hann_window(tile_w, periodic=False, device=device).clamp_min(1e-3)
    return torch.outer(y, x).view(1, 1, tile_h, tile_w)


@torch.no_grad()
def tiled_inference(model: nn.Module, x: Tensor, tile: int = 256, overlap: int = 48) -> Tensor:
    if tile <= 0:
        return simple_inference(model, x)
    _, _, height, width = x.shape
    if height <= tile and width <= tile:
        return simple_inference(model, x)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")

    out = torch.zeros_like(x)
    weight_sum = torch.zeros_like(x[:, :1])
    y_positions = list(range(0, max(height - tile, 0) + 1, stride))
    x_positions = list(range(0, max(width - tile, 0) + 1, stride))
    if y_positions[-1] != height - tile:
        y_positions.append(max(height - tile, 0))
    if x_positions[-1] != width - tile:
        x_positions.append(max(width - tile, 0))

    for top in y_positions:
        for left in x_positions:
            patch = x[:, :, top : top + tile, left : left + tile]
            patch_h, patch_w = patch.shape[-2:]
            pred = simple_inference(model, patch)
            weight = _build_weight(patch_h, patch_w, x.device)
            out[:, :, top : top + patch_h, left : left + patch_w] += pred * weight
            weight_sum[:, :, top : top + patch_h, left : left + patch_w] += weight
    return out / weight_sum.clamp_min(1e-8)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        weight_x = self.sobel_x.repeat(channels, 1, 1, 1)
        weight_y = self.sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(self._gradient(pred), self._gradient(target))


class FrequencyLoss(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(torch.log1p(torch.abs(pred_fft)), torch.log1p(torch.abs(target_fft)))


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.shape[0] <= 1:
            return features.new_tensor(0.0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(features.device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        positives = mask.sum(dim=1)
        valid = positives > 0
        if not valid.any():
            return features.new_tensor(0.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)
        return -mean_log_prob_pos[valid].mean()


def prompt_diversity_loss(prompt_banks: Iterable[Tensor]) -> Tensor:
    losses: List[Tensor] = []
    for bank in prompt_banks:
        prompts = F.normalize(bank.flatten(1), dim=1)
        sim = torch.matmul(prompts, prompts.t())
        off_diag = sim - torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        losses.append(off_diag.square().mean())
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


class RestorationCriterion(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.freq = FrequencyLoss()
        self.supcon = SupConLoss()
        self.weights = {
            "edge": float(cfg.get("lambda_edge", 0.05)),
            "freq": float(cfg.get("lambda_freq", 0.05)),
            "task": float(cfg.get("lambda_task", 0.05)),
            "supcon": float(cfg.get("lambda_supcon", 0.02)),
            "prompt_div": float(cfg.get("lambda_prompt_div", 0.005)),
        }

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        labels: Tensor,
        aux: Dict[str, Any],
        prompt_banks: Iterable[Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        pred = pred.clamp(0.0, 1.0)
        recon = self.charbonnier(pred, target)
        edge = self.edge(pred, target)
        freq = self.freq(pred, target)
        loss = recon + self.weights["edge"] * edge + self.weights["freq"] * freq

        task = pred.new_tensor(0.0)
        supcon = pred.new_tensor(0.0)
        div = pred.new_tensor(0.0)
        prompt_aux = aux.get("prompt_aux") if isinstance(aux, dict) else None
        if prompt_aux:
            task_logits = torch.stack([item["task_logits"] for item in prompt_aux], dim=0).mean(dim=0)
            task = F.cross_entropy(task_logits, labels)
            loss = loss + self.weights["task"] * task
        if isinstance(aux, dict) and "embedding" in aux:
            supcon = self.supcon(aux["embedding"], labels)
            loss = loss + self.weights["supcon"] * supcon
        prompt_banks = list(prompt_banks)
        if prompt_banks:
            div = prompt_diversity_loss(prompt_banks).to(pred.device)
            loss = loss + self.weights["prompt_div"] * div

        logs = {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "freq": float(freq.detach().cpu()),
            "task": float(task.detach().cpu()),
            "supcon": float(supcon.detach().cpu()),
            "div": float(div.detach().cpu()),
        }
        return loss, logs


# -----------------------------------------------------------------------------
# Model / data / checkpoint
# -----------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> Dict[str, Tensor]:
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.ema.load_state_dict(state_dict)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    module = importlib.import_module(f"models.{model_cfg['name']}")
    return module.build_model(model_cfg)


def model_forward(model: nn.Module, degraded: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        out = model(degraded, return_aux=True)
    except TypeError:
        return model(degraded), {}
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def get_prompt_banks(model: nn.Module) -> List[Tensor]:
    if hasattr(model, "prompt_parameters"):
        return list(model.prompt_parameters())
    return []


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    seed = int(cfg.get("project", {}).get("seed", 3407))
    data_cfg = cfg["data"]
    pairs = scan_train_pairs(data_cfg["root"])
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=float(data_cfg.get("val_ratio", 0.05)), seed=seed)

    train_set = RainSnowTrainDataset(
        train_pairs,
        crop_size=int(data_cfg.get("crop_size", 256)),
        augment_config=data_cfg.get("augment", {}),
        train=True,
    )
    val_set = RainSnowTrainDataset(val_pairs, crop_size=None, augment_config={}, train=False)

    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=True,
        persistent_workers=bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    print(f"Train images: {len(train_set)} | Validation images: {len(val_set)}")
    return train_loader, val_loader


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_psnr: float,
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": cfg,
        },
        path,
    )


def same_label_mixup(
    degraded: Tensor,
    clean: Tensor,
    labels: Tensor,
    aug_cfg: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    cfg = aug_cfg.get("same_label_mixup", {})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return degraded, clean, labels
    alpha = float(cfg.get("alpha", 0.2))
    if alpha <= 0:
        return degraded, clean, labels
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.arange(labels.shape[0], device=labels.device)
    for label in labels.unique():
        mask = torch.where(labels == label)[0]
        if mask.numel() > 1:
            perm = mask[torch.randperm(mask.numel(), device=labels.device)]
            indices[mask] = perm
    degraded = lam * degraded + (1.0 - lam) * degraded[indices]
    clean = lam * clean + (1.0 - lam) * clean[indices]
    return degraded, clean, labels


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    writer: Any,
    epoch: int,
) -> float:
    model.eval()
    psnrs: List[float] = []
    tile = int(cfg.get("validation", {}).get("tile", 256))
    overlap = int(cfg.get("validation", {}).get("overlap", 48))
    log_images_every = int(cfg.get("validation", {}).get("log_images_every", 0))
    logged_image = False

    for batch in tqdm(val_loader, desc="val", leave=False):
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        pred = tiled_inference(model, degraded, tile=tile, overlap=overlap)
        psnrs.append(tensor_psnr(pred, clean).mean().item())
        if writer is not None and log_images_every > 0 and epoch % log_images_every == 0 and not logged_image:
            grid = torch.cat([degraded[0], pred[0].clamp(0, 1), clean[0]], dim=2)
            writer.add_image("val/degraded_pred_clean", grid, epoch)
            logged_image = True
    return sum(psnrs) / len(psnrs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = prepare_config()
    seed_everything(int(cfg.get("project", {}).get("seed", 3407)))
    dirs = create_run_dirs(cfg)
    save_yaml(cfg, dirs["run"] / "config_resolved.yaml")
    save_json(cfg, dirs["run"] / "config_resolved.json")
    print(f"Run directory: {dirs['run']}")

    writer = None
    if cfg.get("logging", {}).get("tensorboard", True):
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard is not installed. Run: pip install tensorboard")
        writer = SummaryWriter(log_dir=str(dirs["logs"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['name']} | parameters: {count_parameters(model):.2f}M")

    criterion = RestorationCriterion(cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"].get("epochs", 300)),
        eta_min=float(cfg["train"].get("min_lr", 1e-6)),
    )
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))

    start_epoch = 1
    best_psnr = 0.0
    resume = str(cfg["train"].get("resume", "") or "")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", 0.0))
        print(f"Resumed from {resume}; next epoch = {start_epoch}")

    epochs = int(cfg["train"].get("epochs", 300))
    accum_steps = int(cfg["train"].get("accum_steps", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    print_every = int(cfg.get("logging", {}).get("print_every", 20))
    val_every = int(cfg.get("validation", {}).get("every", 1))
    ckpt_cfg = cfg.get("checkpoint", {})

    global_step = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_logs: List[Dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            global_step += 1
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            degraded, clean, labels = same_label_mixup(degraded, clean, labels, cfg["data"].get("augment", {}))

            with autocast(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda"):
                pred, aux = model_forward(model, degraded)
                loss, logs = criterion(pred, clean, labels, aux, get_prompt_banks(model))
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()
            should_step = step % accum_steps == 0 or step == len(train_loader)
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            epoch_logs.append(logs)
            if writer is not None:
                writer.add_scalar("train/loss_step", logs["loss"], global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            if step % print_every == 0:
                progress.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        train_logs = average_logs(epoch_logs)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}: {format_logs(train_logs)} | time: {elapsed:.1f}s")
        if writer is not None:
            for key, value in train_logs.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            writer.add_scalar("train/epoch_time_sec", elapsed, epoch)

        val_psnr = None
        if val_every > 0 and epoch % val_every == 0:
            val_psnr = validate(ema.ema, val_loader, device, cfg, writer, epoch)
            print(f"Epoch {epoch}: EMA validation PSNR = {val_psnr:.4f} dB")
            if writer is not None:
                writer.add_scalar("val/psnr_ema", val_psnr, epoch)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                if ckpt_cfg.get("save_best", True):
                    save_checkpoint(dirs["checkpoints"] / "best.pt", model, ema, optimizer, scheduler, epoch, best_psnr, cfg)
                    print(f"Saved best checkpoint: {best_psnr:.4f} dB")

        if ckpt_cfg.get("save_last", True):
            save_checkpoint(dirs["checkpoints"] / "last.pt", model, ema, optimizer, scheduler, epoch, best_psnr, cfg)
        save_every = int(ckpt_cfg.get("save_every", 0) or 0)
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                dirs["checkpoints"] / f"epoch_{epoch:04d}.pt",
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                cfg,
            )

    if writer is not None:
        writer.close()
    print(f"Training done. Best validation PSNR: {best_psnr:.4f} dB")
    print(f"Artifacts saved under: {dirs['run']}")


if __name__ == "__main__":
    main()
