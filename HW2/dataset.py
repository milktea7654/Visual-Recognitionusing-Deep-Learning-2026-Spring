"""
Dataset and augmentation pipeline for DINO digit detection.

Wraps torchvision CocoDetection with bbox-aware augmentations:
affine rotation, perspective warp, random crop, sensor noise,
Gaussian blur, colour jitter, and aspect-ratio-preserving resize.
"""
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

# ── box format conversion (inlined) ─────────────────────────────────────────


def _xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) * .5, (y0 + y1) * .5, x1 - x0, y1 - y0], -1)


# ── single-image resize (keeps aspect ratio) ────────────────────────────────


def _resize_with_boxes(img, tgt, short_edge, max_side=None):
    """Resize so shorter side == *short_edge*; optionally cap longer side."""
    w, h = img.size
    if isinstance(short_edge, (list, tuple)):
        new_h, new_w = short_edge[1], short_edge[0]  # explicit (w,h)
    else:
        mn, mx = (w, h) if w < h else (h, w)
        ratio = short_edge / mn
        if max_side is not None and mx * ratio > max_side:
            ratio = max_side / mx
        new_w, new_h = int(w * ratio), int(h * ratio)

    img = TF.resize(img, (new_h, new_w))
    if tgt is None:
        return img, None

    tgt = tgt.copy()
    rw, rh = new_w / w, new_h / h
    if "boxes" in tgt:
        tgt["boxes"] = tgt["boxes"] * torch.tensor([rw, rh, rw, rh])
    if "area" in tgt:
        tgt["area"] = tgt["area"] * (rw * rh)
    tgt["size"] = torch.tensor([new_h, new_w])
    return img, tgt


# ── box visibility filter ───────────────────────────────────────────────────


def _clip_and_filter(tgt, width, height, vis_thr=0.5):
    """Clip boxes to canvas and drop entries below *vis_thr* visibility."""
    if tgt is None or "boxes" not in tgt or tgt["boxes"].numel() == 0:
        return tgt
    tgt = tgt.copy()
    bx = tgt["boxes"]
    old_area = (bx[:, 2] - bx[:, 0]).clamp(
        min=1e-6) * (bx[:, 3] - bx[:, 1]).clamp(min=1e-6)
    bx[:, 0::2].clamp_(0, width)
    bx[:, 1::2].clamp_(0, height)
    new_area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    keep = (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1]) & (new_area / old_area
                                                            >= vis_thr)
    for field in ("boxes", "labels", "area", "iscrowd"):
        if field in tgt:
            tgt[field] = tgt[field][keep]
    return tgt


# ── augmentation transforms ─────────────────────────────────────────────────


class PhotometricJitter:
    """Random brightness / contrast / saturation / hue."""

    def __init__(self,
                 brightness=.2,
                 contrast=.2,
                 saturation=.2,
                 hue=.05,
                 prob=1.):
        self._jit = torchvision.transforms.ColorJitter(brightness, contrast,
                                                       saturation, hue)
        self._p = prob

    def __call__(self, im, tgt):
        return (self._jit(im), tgt) if random.random() < self._p else (im, tgt)


class ScaleJitter:
    """Pick one of several fixed short-edge sizes and resize."""

    def __init__(self, sizes, max_side=None):
        self._sizes = list(sizes)
        self._ms = max_side

    def __call__(self, im, tgt=None):
        return _resize_with_boxes(im, tgt, random.choice(self._sizes), self._ms)


class ContinuousScale:
    """Resize by a scale drawn uniformly from [lo, hi]."""

    def __init__(self, lo=0.8, hi=1.0):
        self._lo, self._hi = lo, hi

    def __call__(self, im, tgt=None):
        s = random.uniform(self._lo, self._hi)
        nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
        return _resize_with_boxes(im, tgt, (nw, nh))


class AffineRotation:
    """Rotation + translation with scale-up to avoid black borders."""

    def __init__(self, max_deg=10., translate=(.1, .1), vis_thr=.5, prob=1.):
        self._deg = max_deg
        self._t = translate
        self._vis = vis_thr
        self._p = prob

    def __call__(self, im, tgt):
        if random.random() >= self._p:
            return im, tgt
        w, h = im.size
        ang = random.uniform(-self._deg, self._deg)
        dx = random.uniform(-self._t[0], self._t[0]) * w
        dy = random.uniform(-self._t[1], self._t[1]) * h

        # scale up so rotated corners still cover canvas
        rad = math.radians(abs(ang))
        sc = math.cos(rad) + math.sin(rad) * max(h / w, w / h)

        im = TF.affine(im,
                       angle=ang,
                       translate=[int(dx), int(dy)],
                       scale=sc,
                       shear=0,
                       fill=0)

        if tgt is not None and "boxes" in tgt and tgt["boxes"].numel():
            tgt = tgt.copy()
            bx = tgt["boxes"].clone().float()
            ox, oy = w / 2., h / 2.
            xs = torch.stack([bx[:, 0], bx[:, 2], bx[:, 2], bx[:, 0]], 1) - ox
            ys = torch.stack([bx[:, 1], bx[:, 1], bx[:, 3], bx[:, 3]], 1) - oy
            a = math.radians(-ang)
            ca, sa = math.cos(a), math.sin(a)
            rx = sc * (ca * xs - sa * ys) + dx + ox
            ry = sc * (sa * xs + ca * ys) + dy + oy
            tgt["boxes"] = torch.stack([
                rx.min(1).values,
                ry.min(1).values,
                rx.max(1).values,
                ry.max(1).values
            ], 1)
            tgt = _clip_and_filter(tgt, w, h, self._vis)
        return im, tgt


class PerspectiveWarp:
    """Corner-perturbation perspective transform with homography box update."""

    def __init__(self, strength=.1, prob=.5, vis_thr=.5):
        self._str = strength
        self._p = prob
        self._vis = vis_thr

    @staticmethod
    def _solve_H(src_pts, dst_pts):
        rows = []
        for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
            rows.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
            rows.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
        _, _, vt = np.linalg.svd(np.array(rows, dtype=np.float64))
        H = vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def __call__(self, im, tgt):
        if random.random() >= self._p:
            return im, tgt
        w, h = im.size
        d = self._str * min(w, h)

        def jitter(x, y):
            return [x + random.uniform(-d, d), y + random.uniform(-d, d)]

        src = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        dst = [
            jitter(0, 0),
            jitter(w - 1, 0),
            jitter(w - 1, h - 1),
            jitter(0, h - 1)
        ]
        im = TF.perspective(im, src, dst, fill=0)

        if tgt is not None and "boxes" in tgt and tgt["boxes"].numel():
            tgt = tgt.copy()
            H = torch.from_numpy(self._solve_H(np.array(src),
                                               np.array(dst))).float()
            bx = tgt["boxes"].clone().float()
            corners_x = torch.stack([bx[:, 0], bx[:, 2], bx[:, 2], bx[:, 0]], 1)
            corners_y = torch.stack([bx[:, 1], bx[:, 1], bx[:, 3], bx[:, 3]], 1)
            n = bx.shape[0]
            pts = torch.stack([corners_x, corners_y, torch.ones(n, 4)], 2)
            warped = torch.einsum("ij,nkj->nki", H, pts)
            wx = warped[:, :, 0] / warped[:, :, 2]
            wy = warped[:, :, 1] / warped[:, :, 2]
            tgt["boxes"] = torch.stack([
                wx.min(1).values,
                wy.min(1).values,
                wx.max(1).values,
                wy.max(1).values
            ], 1)
            tgt = _clip_and_filter(tgt, w, h, self._vis)
        return im, tgt


class BoxAwareCrop:
    """Random crop that keeps boxes above a visibility threshold."""

    def __init__(self, scale_range=(.6, 1.), prob=.5, vis_thr=.5):
        self._sr = scale_range
        self._p = prob
        self._vis = vis_thr

    def __call__(self, im, tgt):
        if random.random() >= self._p:
            return im, tgt
        w, h = im.size
        s = random.uniform(*self._sr)
        cw, ch = max(1, int(w * s)), max(1, int(h * s))
        x0 = random.randint(0, max(0, w - cw))
        y0 = random.randint(0, max(0, h - ch))
        im = TF.crop(im, y0, x0, ch, cw)
        if tgt is not None and "boxes" in tgt and tgt["boxes"].numel():
            tgt = tgt.copy()
            tgt["boxes"] = tgt["boxes"].clone()
            tgt["boxes"][:, 0::2] -= x0
            tgt["boxes"][:, 1::2] -= y0
            tgt = _clip_and_filter(tgt, cw, ch, self._vis)
            tgt["size"] = torch.tensor([ch, cw])
        return im, tgt


class SensorNoise:
    """Synthetic ISO-like noise in HSV space."""

    def __init__(self, prob=.2, strength=.05, hue_shift=.05):
        self._p = prob
        self._s = strength
        self._h = hue_shift

    def __call__(self, im, tgt):
        if random.random() >= self._p:
            return im, tgt
        arr = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.
        arr[..., 0] = (arr[..., 0] + np.random.uniform(-self._h, self._h)) % 1.
        noise = 1. + np.random.randn(*arr.shape[:2]) * self._s
        arr[..., 1:] = np.clip(arr[..., 1:] * noise[..., None], 0, 1)
        im = Image.fromarray((arr * 255).astype(np.uint8), "HSV").convert("RGB")
        return im, tgt


class BlurAugment:
    """Random Gaussian blur."""

    def __init__(self, sigma=(.1, 2.), prob=.5):
        self._sig = sigma
        self._p = prob

    def __call__(self, im, tgt):
        if random.random() < self._p:
            im = im.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(*self._sig)))
        return im, tgt


class ConvertToTensor:

    def __call__(self, im, tgt):
        return TF.to_tensor(im), tgt


class NormalizeBoxes:
    """ImageNet-normalise image; convert boxes to normalised cxcywh."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._m, self._s = mean, std

    def __call__(self, im, tgt=None):
        im = TF.normalize(im, self._m, self._s)
        if tgt is None:
            return im, None
        tgt = tgt.copy()
        h, w = im.shape[-2:]
        if "boxes" in tgt:
            tgt["boxes"] = _xyxy_to_cxcywh(tgt["boxes"]) / torch.tensor(
                [w, h, w, h], dtype=torch.float32)
        return im, tgt


class AugPipeline:
    """Chain of (image, target) transforms."""

    def __init__(self, ops):
        self._ops = ops

    def __call__(self, im, tgt):
        for op in self._ops:
            im, tgt = op(im, tgt)
        return im, tgt


# ── COCO annotation converter ───────────────────────────────────────────────


class CocoBoxConverter:
    """Extract boxes / labels / area from raw COCO-style annotation list."""

    def __call__(self, image, raw_target):
        w, h = image.size
        img_id = raw_target["image_id"]
        objs = [
            o for o in raw_target["annotations"] if o.get("iscrowd", 0) == 0
        ]

        boxes = torch.as_tensor([o["bbox"] for o in objs],
                                dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh → xyxy
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)

        labels = torch.tensor([o["category_id"] for o in objs],
                              dtype=torch.int64)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        return image, {
            "boxes": boxes[valid],
            "labels": labels[valid],
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor([o["area"] for o in objs])[valid],
            "iscrowd": torch.tensor([o.get("iscrowd", 0) for o in objs])[valid],
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([h, w]),
        }


# ── dataset wrapper ─────────────────────────────────────────────────────────


class DigitCocoDataset(torchvision.datasets.CocoDetection):

    def __init__(self, img_dir, ann_path, transforms):
        super().__init__(img_dir, ann_path)
        self._aug = transforms
        self._conv = CocoBoxConverter()

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        img, tgt = self._conv(img, {
            "image_id": self.ids[idx],
            "annotations": ann
        })
        if self._aug is not None:
            img, tgt = self._aug(img, tgt)
        return img, tgt


# ── builder helpers ──────────────────────────────────────────────────────────


def build_augmentations(split, cfg=None):
    """Build augmentation pipeline for *split* ('train' | 'val')."""
    c = cfg or {}
    tail = AugPipeline([ConvertToTensor(), NormalizeBoxes()])

    if split == "train":
        se = c.get("train_short_edge", 320)
        ms = c.get("train_max_size", 640)
        mv = c.get("aug_min_visibility", .5)
        ops = [
            PhotometricJitter(prob=c.get("aug_color_jitter_prob", 1.)),
            BlurAugment(sigma=tuple(c.get("aug_blur_sigma", [.1, 2.])),
                        prob=c.get("aug_blur_prob", .5)),
        ]
        if c.get("aug_use_iso_noise", False):
            ops.append(
                SensorNoise(prob=c.get("aug_iso_noise_prob", .2),
                            strength=c.get("aug_iso_noise_intensity", .05)))
        ops += [
            AffineRotation(max_deg=c.get("aug_rotation_degrees", 10.),
                           translate=tuple(c.get("aug_translate_xy", [.1, .1])),
                           vis_thr=mv,
                           prob=c.get("aug_rotation_prob", 1.)),
            PerspectiveWarp(strength=c.get("aug_perspective_scale", .1),
                            prob=c.get("aug_perspective_prob", .5),
                            vis_thr=mv),
            BoxAwareCrop(scale_range=tuple(c.get("aug_crop_scale", [.6, 1.])),
                         prob=c.get("aug_crop_prob", .5),
                         vis_thr=mv),
            ScaleJitter([se], max_side=ms),
            ContinuousScale(c.get("aug_scale_min", .9),
                            c.get("aug_scale_max", 1.)),
            tail,
        ]
        return AugPipeline(ops)

    if split == "val":
        return AugPipeline([
            ScaleJitter([c.get("val_short_edge", 320)],
                        max_side=c.get("val_max_size", 640)),
            tail,
        ])

    raise ValueError(f"Unknown split: {split}")


def build_dataset(split, cfg):
    root = Path(cfg["data_path"])
    paths = {
        "train": (root / "train", root / cfg["train_json"]),
        "val": (root / "valid", root / cfg["valid_json"]),
    }
    img_dir, ann_path = paths[split]
    return DigitCocoDataset(img_dir, ann_path, build_augmentations(split, cfg))


def get_coco_api(dataset):
    """Unwrap dataset wrappers to reach the underlying COCO object."""
    ds = dataset
    for _ in range(10):
        if isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        else:
            break
    return getattr(ds, "coco", None)
