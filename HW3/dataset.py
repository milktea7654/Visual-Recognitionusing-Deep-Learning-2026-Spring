"""
dataset.py – Cell instance segmentation dataset with heavy augmentation.

Augmentations:
  * Random crop / pad to CROP_SIZE (preserves native scale of small images;
    random patch from large images).
  * Albumentations heavy aug (rotate/flip/colour/elastic/noise) when training.
  * Copy-paste augmentation: paste random instances from other train images
    into background regions.

Returns torchvision-format detection target dict:
    {boxes, labels, masks, image_id, area, iscrowd}
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes

import config

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:
    _HAS_ALB = False


CLASS_NAMES = ["class1", "class2", "class3", "class4"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_image_tif(path: Path) -> np.ndarray:
    """Load a TIFF as uint8 (H, W, 3) RGB array."""
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


def _load_mask_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = arr[0]
    return arr.astype(np.int32)


# ---------------------------------------------------------------------------
# Albumentations pipeline (operates on numpy + masks list)
# ---------------------------------------------------------------------------


def _build_alb_pipeline() -> "A.Compose":
    transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=45,
            border_mode=0, p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    ]
    # HED stain augmentation: simulates H&E staining variation across slides
    # Requires albumentations >= 1.3; gracefully skipped on older versions
    if hasattr(A, "HEDShift"):
        transforms.append(A.HEDShift(
            haematoxylin_bias_range=(-0.1, 0.1),
            eosin_bias_range=(-0.1, 0.1),
            dab_bias_range=(-0.1, 0.1),
            p=0.3,
        ))
    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CellInstanceDataset(data.Dataset):
    """Cell instance-segmentation dataset compatible with torchvision MaskRCNN."""

    def __init__(
        self,
        root: str,
        sample_ids: List[str],
        train: bool = True,
        crop_size: int = config.CROP_SIZE,
    ) -> None:
        self.root = Path(root)
        self.sample_ids = list(sample_ids)
        self.train = train
        self.crop_size = crop_size

        self.use_alb = train and config.USE_ALBUMENTATIONS and _HAS_ALB
        self.alb = _build_alb_pipeline() if self.use_alb else None

        self.use_copy_paste = train and config.USE_COPY_PASTE

        # Pre-compute majority class per sample for class-balanced sampling
        self._majority_class = self._compute_majority_classes()

    # -------------------------------------------------------------------
    # Sample loading
    # -------------------------------------------------------------------

    def _load_sample(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_dir = self.root / sample_id
        img = _load_image_tif(sample_dir / "image.tif")
        H, W = img.shape[:2]

        masks: List[np.ndarray] = []
        labels: List[int] = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES, start=1):
            mp = sample_dir / f"{cls_name}.tif"
            if not mp.exists():
                continue
            mask_arr = _load_mask_tif(mp)
            if mask_arr.shape != (H, W):
                continue
            inst_ids = np.unique(mask_arr)
            inst_ids = inst_ids[inst_ids != 0]
            for iid in inst_ids:
                bin_mask = (mask_arr == iid).astype(np.uint8)
                if bin_mask.sum() == 0:
                    continue
                masks.append(bin_mask)
                labels.append(cls_idx)

        masks_arr = np.stack(masks, axis=0) if masks else np.zeros((0, H, W), np.uint8)
        labels_arr = np.array(labels, dtype=np.int64)
        return img, masks_arr, labels_arr

    def _compute_majority_classes(self) -> List[int]:
        """For class-balanced sampling: dominant class id per sample."""
        out: List[int] = []
        for sid in self.sample_ids:
            sample_dir = self.root / sid
            counts = [0] * (config.NUM_CLASSES + 1)
            for cls_idx, cls_name in enumerate(CLASS_NAMES, start=1):
                if (sample_dir / f"{cls_name}.tif").exists():
                    counts[cls_idx] += 1
            out.append(int(np.argmax(counts)) if max(counts) > 0 else 1)
        return out

    @property
    def majority_classes(self) -> List[int]:
        return self._majority_class

    # -------------------------------------------------------------------
    # Patch crop / pad
    # -------------------------------------------------------------------

    def _crop_or_pad_np(
        self,
        img: np.ndarray,        # (H, W, 3) uint8
        masks: np.ndarray,      # (N, H, W) uint8
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape[:2]
        cs = self.crop_size

        pad_h = max(0, cs - H)
        pad_w = max(0, cs - W)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            if masks.shape[0] > 0:
                masks = np.pad(masks, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            else:
                masks = np.zeros((0, H + pad_h, W + pad_w), np.uint8)
            H, W = H + pad_h, W + pad_w

        if H > cs or W > cs:
            if self.train:
                top = random.randint(0, H - cs)
                left = random.randint(0, W - cs)
            else:
                top = (H - cs) // 2
                left = (W - cs) // 2
            img = img[top:top + cs, left:left + cs]
            if masks.shape[0] > 0:
                masks = masks[:, top:top + cs, left:left + cs]
            else:
                masks = np.zeros((0, cs, cs), np.uint8)

        return img, masks

    # -------------------------------------------------------------------
    # Albumentations wrap
    # -------------------------------------------------------------------

    def _apply_alb(
        self, img: np.ndarray, masks: np.ndarray, labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_alb or masks.shape[0] == 0:
            return img, masks, labels
        # Albumentations needs masks as list of (H, W)
        masks_list = [m for m in masks]
        out = self.alb(image=img, masks=masks_list)
        img2 = out["image"]
        masks2 = np.stack(out["masks"], axis=0).astype(np.uint8) if out["masks"] else np.zeros((0,) + img2.shape[:2], np.uint8)
        return img2, masks2, labels

    # -------------------------------------------------------------------
    # Copy-paste aug
    # -------------------------------------------------------------------

    def _copy_paste(
        self, img: np.ndarray, masks: np.ndarray, labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Paste up to N instances from another random sample.
        Both source instance pixels and mask are pasted; label = source label.
        """
        if random.random() > config.COPY_PASTE_PROB:
            return img, masks, labels

        donor_id = random.choice(self.sample_ids)
        d_img, d_masks, d_labels = self._load_sample(donor_id)
        if d_masks.shape[0] == 0:
            return img, masks, labels

        d_img, d_masks = self._crop_or_pad_np(d_img, d_masks)
        # filter empty masks after crop
        keep = d_masks.reshape(d_masks.shape[0], -1).any(axis=1)
        d_masks = d_masks[keep]
        d_labels = d_labels[keep] if d_labels.shape[0] == keep.shape[0] else d_labels

        if d_masks.shape[0] == 0:
            return img, masks, labels

        n_paste = random.randint(1, min(config.COPY_PASTE_MAX_OBJECTS, d_masks.shape[0]))
        idx = np.random.choice(d_masks.shape[0], size=n_paste, replace=False)

        new_masks = [m for m in masks]
        new_labels = list(labels.tolist())
        out_img = img.copy()

        for i in idx:
            m = d_masks[i].astype(bool)
            if not m.any():
                continue
            out_img[m] = d_img[m]
            new_masks.append(m.astype(np.uint8))
            new_labels.append(int(d_labels[i]))

        new_masks_arr = np.stack(new_masks, axis=0) if new_masks else np.zeros_like(masks)
        new_labels_arr = np.array(new_labels, dtype=np.int64)

        # Occlusion: subtract pasted pixels from earlier masks
        if new_masks_arr.shape[0] > n_paste:
            paste_union = np.zeros(out_img.shape[:2], bool)
            for i in idx:
                paste_union |= d_masks[i].astype(bool)
            for j in range(new_masks_arr.shape[0] - n_paste):
                new_masks_arr[j][paste_union] = 0

        return out_img, new_masks_arr, new_labels_arr

    # -------------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_id = self.sample_ids[idx]
        img_np, masks_np, labels_np = self._load_sample(sample_id)

        # patch crop/pad on numpy
        img_np, masks_np = self._crop_or_pad_np(img_np, masks_np)

        # filter empty masks after crop
        if masks_np.shape[0] > 0:
            keep = masks_np.reshape(masks_np.shape[0], -1).any(axis=1)
            masks_np = masks_np[keep]
            labels_np = labels_np[keep]

        # copy-paste
        if self.use_copy_paste:
            img_np, masks_np, labels_np = self._copy_paste(img_np, masks_np, labels_np)

        # albumentations heavy aug
        img_np, masks_np, labels_np = self._apply_alb(img_np, masks_np, labels_np)

        # to tensor
        img = TF.to_tensor(img_np)                            # (3, H, W)
        masks = torch.from_numpy(masks_np).to(torch.uint8)    # (N, H, W)
        labels = torch.from_numpy(labels_np).to(torch.int64)

        # filter empty / invalid instances
        if masks.numel() > 0:
            keep = masks.flatten(1).any(dim=1).bool()
            masks = masks[keep]
            labels = labels[keep]

        if masks.shape[0] > 0:
            boxes = masks_to_boxes(masks)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area":     area,
            "iscrowd":  iscrowd,
        }
        return img, target

    def __len__(self) -> int:
        return len(self.sample_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_train_val_split(
    root: str = config.DATA_ROOT,
    val_ratio: float = config.VAL_SPLIT,
    seed: int = config.SEED,
) -> Tuple[List[str], List[str]]:
    root_path = Path(root)
    ids = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_class_balanced_weights(majority_classes: List[int]) -> List[float]:
    """Inverse-frequency sample weights for WeightedRandomSampler."""
    counts = np.bincount(majority_classes, minlength=config.NUM_CLASSES + 1).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    return [float(inv[c]) for c in majority_classes]
