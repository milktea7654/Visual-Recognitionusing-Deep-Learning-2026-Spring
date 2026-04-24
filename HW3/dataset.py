"""
dataset.py – PyTorch Dataset for HW3 medical cell instance segmentation.

Directory layout expected:
    data/train/<uuid>/image.tif          – RGB or greyscale image
    data/train/<uuid>/class{1,2,3,4}.tif – instance mask (uint16 / uint8);
                                           each unique non-zero pixel value
                                           is one instance.
    data/test_release/<uuid>.tif         – test images (no masks)

Each class mask file is optional; a folder may have only a subset of classes.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = 4  # foreground classes; index 0 = background internally


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tif(path: str) -> np.ndarray:
    """
    Load a .tif file and return a NumPy array (H, W) or (H, W, C).

    Uses ``tifffile`` instead of PIL to handle scientific/medical TIFF
    variants (LZW compression, BigTIFF, multi-page, uint16, etc.).
    Multi-page TIFFs are squeezed to the first page.
    """
    arr = tifffile.imread(path)
    # tifffile may return (pages, H, W) for multi-page TIFFs – take page 0.
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = arr[0]
    return arr


def _parse_instance_mask(
    mask_arr: np.ndarray,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert a single-channel instance mask into per-instance binary masks.

    Uses PyTorch tensor broadcasting (backed by C++ multi-threading) which
    is faster than pure NumPy for large arrays with many instances.

    Args:
        mask_arr: (H, W) integer array where each unique non-zero value is
                  one cell instance.

    Returns:
        masks: list of boolean (H, W) arrays, one per instance.
        instance_ids: the original pixel values (for debugging).
    """
    instance_ids = np.unique(mask_arr)
    instance_ids = instance_ids[instance_ids != 0]
    if len(instance_ids) == 0:
        return [], instance_ids.tolist()
    mask_tensor = torch.from_numpy(mask_arr)
    ids_tensor = torch.from_numpy(instance_ids)
    # (N, H, W) bool – single broadcast op via PyTorch C++ backend
    masks_bool = mask_tensor.unsqueeze(0) == ids_tensor.unsqueeze(1).unsqueeze(2)
    return list(masks_bool.numpy()), instance_ids.tolist()


def _to_rgb_tensor(img_arr: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image (H, W) or (H, W, C) to a float32 (3, H, W) tensor
    normalised to [0, 1].
    """
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[2] == 1:
        img_arr = np.concatenate([img_arr] * 3, axis=-1)
    elif img_arr.shape[2] > 3:
        img_arr = img_arr[:, :, :3]

    # Normalise to [0, 1] regardless of bit-depth (8-bit or 16-bit TIFFs).
    if np.issubdtype(img_arr.dtype, np.integer):
        max_val = float(np.iinfo(img_arr.dtype).max)
    else:
        max_val = float(img_arr.max()) if img_arr.max() > 0 else 1.0
    img_arr = img_arr.astype(np.float32) / max_val
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, H, W)
    return tensor


# ---------------------------------------------------------------------------
# Augmentation helpers (geometry-consistent on image + masks)
# ---------------------------------------------------------------------------


def _random_horizontal_flip(
    image: torch.Tensor,
    masks: Optional[torch.Tensor],
    p: float = 0.5,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if random.random() < p:
        image = TF.hflip(image)
        if masks is not None:
            masks = torch.flip(masks, dims=[-1])
    return image, masks


def _random_vertical_flip(
    image: torch.Tensor,
    masks: Optional[torch.Tensor],
    p: float = 0.5,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if random.random() < p:
        image = TF.vflip(image)
        if masks is not None:
            masks = torch.flip(masks, dims=[-2])
    return image, masks


def _random_crop(
    image: torch.Tensor,
    masks: Optional[torch.Tensor],
    min_scale: float = 0.7,
    max_scale: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Random crop with scale in [min_scale, max_scale] of the original size,
    then resize back to original dimensions.  Keeps all masks aligned.
    """
    _, h, w = image.shape
    scale = random.uniform(min_scale, max_scale)
    ch = int(h * scale)
    cw = int(w * scale)
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)

    image = TF.crop(image, top, left, ch, cw)
    image = TF.resize(image, [h, w], antialias=True)

    if masks is not None and masks.shape[0] > 0:
        masks = TF.crop(masks, top, left, ch, cw)
        # Use nearest neighbor to preserve integer instance IDs
        masks = TF.resize(
            masks.unsqueeze(1).float(),
            [h, w],
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(1).to(masks.dtype)

    return image, masks


def _color_jitter(
    image: torch.Tensor,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.2,
    hue: float = 0.05,
) -> torch.Tensor:
    """Apply colour jitter to the image tensor only."""
    from torchvision import transforms as T

    jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    return jitter(image)


def _gaussian_blur(image: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    if random.random() < p:
        sigma = random.uniform(0.1, 1.5)
        image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
    return image


def _normalize(image: torch.Tensor) -> torch.Tensor:
    """ImageNet mean/std normalisation."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CellInstanceDataset(data.Dataset):
    """
    Loads medical cell images and their instance segmentation masks.

    Args:
        root_dir:         Path to ``data/train``.
        augment:          Apply training augmentations when True.
        crop_size:        If image is larger than this, randomly crop to
                          ``crop_size × crop_size`` (never resize down).
                          For images smaller than crop_size, pad up to the
                          nearest size_divisibility multiple instead.
        size_divisibility: Pad dims to be a multiple of this (32 for ConvNeXt).
        min_size:         Minimum output dim; small images padded to this.
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = True,
        crop_size: Optional[int] = 768,
        size_divisibility: int = 32,
        min_size: Optional[int] = 256,
        # Legacy aliases (ignored; kept for call-site compatibility)
        max_size: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.crop_size = crop_size
        self.size_divisibility = size_divisibility
        self.min_size = min_size
        # Legacy aliases
        self.max_size = crop_size

        self.sample_dirs: List[Path] = sorted(
            [p for p in self.root_dir.iterdir() if p.is_dir()]
        )
        # In-memory tensor cache: key = folder name, value = (image, masks_stacked)
        # First epoch reads from disk; subsequent epochs hit this dict (~0 ms I/O).
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.sample_dirs)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_sample(
        self, sample_dir: Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image:         (3, H, W) float32 tensor in [0, 1].
            masks_stacked: (4, H, W) int32 tensor – raw instance IDs per class
                           (class index = channel, 0 = background).
        """
        cache_key = sample_dir.name
        if cache_key in self.cache:
            # .clone() prevents augmentation from corrupting the cached tensors
            img_c, msk_c = self.cache[cache_key]
            return img_c.clone(), msk_c.clone()

        img_arr = _load_tif(str(sample_dir / "image.tif"))
        image = _to_rgb_tensor(img_arr)

        _, h, w = image.shape
        masks_stacked = torch.zeros((NUM_CLASSES, h, w), dtype=torch.int32)

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask_path = sample_dir / f"{cls_name}.tif"
            if not mask_path.exists():
                continue
            mask_arr = _load_tif(str(mask_path))
            masks_stacked[cls_idx] = torch.from_numpy(mask_arr.astype(np.int32))

        # Write to cache after first load
        self.cache[cache_key] = (image.clone(), masks_stacked.clone())
        return image, masks_stacked

    # ------------------------------------------------------------------
    # Augmentation pipeline
    # ------------------------------------------------------------------

    def _augment(
        self,
        image: torch.Tensor,
        masks_stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, masks_stacked = _random_horizontal_flip(image, masks_stacked)
        image, masks_stacked = _random_vertical_flip(image, masks_stacked)
        image, masks_stacked = _random_crop(image, masks_stacked)
        image = _color_jitter(image)
        image = _gaussian_blur(image)
        return image, masks_stacked

    # ------------------------------------------------------------------
    # Resize + pad / crop (golden path)
    # ------------------------------------------------------------------

    def _resize_and_pad(
        self,
        image: torch.Tensor,
        masks_stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sizing strategy:
        • If image is LARGER than crop_size in either dimension:
              Randomly crop a crop_size × crop_size window (no resize).
              This preserves physical cell size so small cells are never
              shrunk to sub-pixel features after the 32× stride.
        • If image is SMALLER than crop_size:
              Zero-pad to the nearest size_divisibility multiple ≥ min_size.
        Operates on the compact (4, H, W) int32 stacked mask.
        """
        _, h, w = image.shape
        d = self.size_divisibility

        if self.crop_size is not None and (h > self.crop_size or w > self.crop_size):
            # ---------- random crop path ----------
            ch = min(h, self.crop_size)
            cw = min(w, self.crop_size)
            top  = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            image        = TF.crop(image,        top, left, ch, cw)
            masks_stacked = TF.crop(masks_stacked, top, left, ch, cw)
            h, w = ch, cw

        # Pad to size_divisibility and min_size (no resize)
        target_h = math.ceil(h / d) * d
        target_w = math.ceil(w / d) * d
        if self.min_size is not None:
            target_h = max(target_h, self.min_size)
            target_w = max(target_w, self.min_size)
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            image         = F.pad(image,         (0, pad_w, 0, pad_h), value=0.0)
            masks_stacked = F.pad(masks_stacked, (0, pad_w, 0, pad_h), value=0)

        return image, masks_stacked

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict:
        sample_dir = self.sample_dirs[idx]
        image, masks_stacked = self._load_sample(sample_dir)

        if self.augment:
            image, masks_stacked = self._augment(image, masks_stacked)

        image, masks_stacked = self._resize_and_pad(image, masks_stacked)
        image = _normalize(image)

        # --- Deferred unpacking: (4, H, W) int32 → (N, H, W) bool ---
        # Only happens here, after all resize/augment ops are done.
        all_bool_masks: List[torch.Tensor] = []
        all_labels: List[int] = []
        for cls_idx in range(NUM_CLASSES):
            cls_mask = masks_stacked[cls_idx]          # (H, W) int32
            inst_ids = cls_mask.unique()
            inst_ids = inst_ids[inst_ids != 0]
            if len(inst_ids) > 0:
                # (N, H, W) bool via vectorised comparison
                bool_masks = cls_mask.unsqueeze(0) == inst_ids.unsqueeze(1).unsqueeze(2)
                all_bool_masks.extend(list(bool_masks))
                all_labels.extend([cls_idx + 1] * len(inst_ids))  # 1-indexed

        if all_bool_masks:
            masks_tensor = torch.stack(all_bool_masks)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        else:
            _, h, w = image.shape
            masks_tensor = torch.zeros((0, h, w), dtype=torch.bool)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,               # (3, H, W) float32, normalised
            "masks": masks_tensor,        # (N, H, W) bool
            "labels": labels_tensor,      # (N,) int64, 1-indexed class ids
            "image_id": sample_dir.name,  # folder UUID string
        }


# ---------------------------------------------------------------------------
# Test-set Dataset (no masks)
# ---------------------------------------------------------------------------


class CellTestDataset(data.Dataset):
    """
    Loads test images from ``data/test_release/``.

    Args:
        root_dir:         Path to ``data/test_release``.
        max_size:         Cap long edge (scale down only).
        min_size:         Minimum dim; pad if smaller.
        size_divisibility: Pad to multiple of this value.
    """

    def __init__(
        self,
        root_dir: str,
        max_size: Optional[int] = 768,
        min_size: Optional[int] = 256,
        size_divisibility: int = 32,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.max_size = max_size
        self.min_size = min_size
        self.size_divisibility = size_divisibility
        self.image_paths: List[Path] = sorted(self.root_dir.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.image_paths[idx]
        img_arr = _load_tif(str(path))
        image = _to_rgb_tensor(img_arr)

        _, h, w = image.shape
        # Scale down only if long edge exceeds max_size
        if self.max_size is not None and max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            image = TF.resize(image, [new_h, new_w], antialias=True)
            h, w = new_h, new_w
        # Pad to size_divisibility and min_size
        d = self.size_divisibility
        target_h = math.ceil(h / d) * d
        target_w = math.ceil(w / d) * d
        if self.min_size is not None:
            target_h = max(target_h, self.min_size)
            target_w = max(target_w, self.min_size)
        pad_h, pad_w = target_h - h, target_w - w
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)

        image = _normalize(image)

        return {
            "image": image,
            "file_name": path.name,
            "orig_h": img_arr.shape[0],
            "orig_w": img_arr.shape[1],
            "padded_h": image.shape[1],
            "padded_w": image.shape[2],
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate that pads variable-length mask tensors.

    Returns a dict with:
        images:  (B, 3, H, W) float32 – padded to max H×W in batch.
        targets: list of per-image dicts {"masks": (N,H,W), "labels": (N,)}.
        image_ids: list of str.
    """
    images = []
    targets = []
    image_ids = []

    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    for item in batch:
        img = item["image"]
        _, h, w = img.shape
        # Pad bottom-right with zero (after normalisation ≈ -mean/std)
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F

            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)

        images.append(img)

        if "masks" in item:
            masks = item["masks"]  # (N, h, w)
            if pad_h > 0 or pad_w > 0:
                import torch.nn.functional as F

                masks = F.pad(
                    masks.float(), (0, pad_w, 0, pad_h), value=0.0
                ).bool()
            targets.append(
                {"masks": masks, "labels": item["labels"]}
            )
            image_ids.append(item["image_id"])

    return {
        "images": torch.stack(images),
        "targets": targets,
        "image_ids": image_ids,
    }
