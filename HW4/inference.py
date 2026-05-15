"""Generate pred.npz and pred.zip for HW4 submission."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RainSnowTestDataset


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
        return yaml.safe_load(handle) or {}


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
    parser = argparse.ArgumentParser(description="HW4 inference.")
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--data_root", type=str, default="", help="Override data.root.")
    parser.add_argument("--checkpoint", type=str, default="", help="Override inference.checkpoint.")
    parser.add_argument("--output_npz", type=str, default="", help="Override inference.output_npz.")
    parser.add_argument("--output_zip", type=str, default="", help="Override inference.output_zip.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted keys.")
    return parser.parse_args()


def prepare_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        set_by_dotted_key(cfg, "data.root", args.data_root)
    if args.checkpoint:
        set_by_dotted_key(cfg, "inference.checkpoint", args.checkpoint)
    if args.output_npz:
        set_by_dotted_key(cfg, "inference.output_npz", args.output_npz)
    if args.output_zip:
        set_by_dotted_key(cfg, "inference.output_zip", args.output_zip)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(cfg, key, yaml.safe_load(raw_value))
    return cfg


# -----------------------------------------------------------------------------
# Utilities
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


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    module = importlib.import_module(f"models.{model_cfg['name']}")
    return module.build_model(model_cfg)


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


def _tta_transforms(x: Tensor) -> List[Tensor]:
    return [
        x,
        torch.flip(x, dims=[-1]),
        torch.flip(x, dims=[-2]),
        torch.flip(x, dims=[-2, -1]),
        x.transpose(-2, -1),
        torch.flip(x.transpose(-2, -1), dims=[-1]),
        torch.flip(x.transpose(-2, -1), dims=[-2]),
        torch.flip(x.transpose(-2, -1), dims=[-2, -1]),
    ]


def _tta_inverse(y: Tensor, index: int) -> Tensor:
    if index == 0:
        return y
    if index == 1:
        return torch.flip(y, dims=[-1])
    if index == 2:
        return torch.flip(y, dims=[-2])
    if index == 3:
        return torch.flip(y, dims=[-2, -1])
    if index == 4:
        return y.transpose(-2, -1)
    if index == 5:
        return torch.flip(y, dims=[-1]).transpose(-2, -1)
    if index == 6:
        return torch.flip(y, dims=[-2]).transpose(-2, -1)
    if index == 7:
        return torch.flip(y, dims=[-2, -1]).transpose(-2, -1)
    raise ValueError(index)


@torch.no_grad()
def tta_inference(model: nn.Module, x: Tensor, tile: int = 256, overlap: int = 48) -> Tensor:
    outputs = []
    for idx, aug in enumerate(_tta_transforms(x)):
        pred = tiled_inference(model, aug, tile=tile, overlap=overlap)
        outputs.append(_tta_inverse(pred, idx))
    return torch.stack(outputs, dim=0).mean(dim=0)


def tensor_to_uint8_chw(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().float().clamp(0.0, 1.0).cpu()
    return (tensor.squeeze(0).numpy() * 255.0).round().astype(np.uint8)


def load_weights(checkpoint_path: str | Path, model: nn.Module, device: torch.device, use_ema: bool) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    key = "ema" if use_ema and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[key], strict=True)
    return checkpoint


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = prepare_config()
    seed_everything(int(cfg.get("project", {}).get("seed", 3407)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = cfg["inference"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "config" in checkpoint and "model" in checkpoint["config"]:
        cfg["model"] = checkpoint["config"]["model"]

    model = build_model(cfg).to(device)
    use_ema = bool(cfg["inference"].get("use_ema", True))
    load_weights(checkpoint_path, model, device, use_ema=use_ema)
    model.eval()

    dataset = RainSnowTestDataset(cfg["data"]["root"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["inference"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["inference"].get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )

    tile = int(cfg["inference"].get("tile", 256))
    overlap = int(cfg["inference"].get("overlap", 48))
    use_tta = bool(cfg["inference"].get("tta", True))
    results: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            degraded = batch["degraded"].to(device, non_blocking=True)
            filenames = batch["filename"]
            pred = tta_inference(model, degraded, tile=tile, overlap=overlap) if use_tta else tiled_inference(model, degraded, tile=tile, overlap=overlap)
            for idx, filename in enumerate(filenames):
                results[filename] = tensor_to_uint8_chw(pred[idx : idx + 1])

    output_npz = Path(cfg["inference"].get("output_npz", "outputs/pred.npz"))
    mkdir(output_npz.parent)
    np.savez_compressed(output_npz, **results)
    print(f"Saved {len(results)} restored images to {output_npz}")

    output_zip = cfg["inference"].get("output_zip", "")
    if output_zip:
        output_zip = Path(output_zip)
        mkdir(output_zip.parent)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_npz, arcname="pred.npz")
        print(f"Saved submission zip to {output_zip}. It contains pred.npz at the root.")


if __name__ == "__main__":
    main()
