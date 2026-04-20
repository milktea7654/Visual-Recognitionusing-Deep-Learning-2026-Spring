"""Dataset and transforms for COCO-format digit detection."""
import os
import json
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Custom transforms (image + target co-transform)
# ---------------------------------------------------------------------------

def _resize(image, target, size, max_size=None):
    """Resize image so the shorter side equals *size*; cap longer side at *max_size*."""
    w, h = image.size  # PIL: (width, height)

    if max_size is not None:
        min_orig = min(h, w)
        max_orig = max(h, w)
        if max_orig / min_orig * size > max_size:
            size = int(round(max_size * min_orig / max_orig))

    # Already the right size?
    if (w <= h and w == size) or (h <= w and h == size):
        return image, target

    if w < h:
        ow, oh = size, int(round(size * h / w))
    else:
        oh, ow = size, int(round(size * w / h))

    image = TF.resize(image, (oh, ow))

    if "boxes" in target and len(target["boxes"]) > 0:
        ratio_w = ow / w
        ratio_h = oh / h
        boxes = target["boxes"].clone()
        boxes = boxes * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        target["boxes"] = boxes

    target["size"] = torch.tensor([oh, ow])
    return image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.choice(self.sizes)
        return _resize(image, target, size, self.max_size)


class Resize:
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image, target):
        return _resize(image, target, self.size, self.max_size)


class ToTensor:
    def __call__(self, image, target):
        return TF.to_tensor(image), target


class Normalize:
    """Normalize image and convert boxes from xyxy pixel to cxcywh normalized."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, self.mean, self.std)

        if "boxes" in target and len(target["boxes"]) > 0:
            _, h, w = image.shape
            boxes = target["boxes"]  # xyxy absolute
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
            bw = (boxes[:, 2] - boxes[:, 0]) / w
            bh = (boxes[:, 3] - boxes[:, 1]) / h
            target["boxes"] = torch.stack([cx, cy, bw, bh], dim=-1)

        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter:
    """Wrapper around torchvision ColorJitter that works with (image, target)."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        return self.jitter(image), target


class RandomErasing:
    """Random erasing on tensor image; works with (image, target)."""

    def __init__(self, p=0.25, scale=(0.01, 0.05)):
        self.eraser = T.RandomErasing(p=p, scale=scale)

    def __call__(self, image, target):
        return self.eraser(image), target


class RandomGaussianBlur:
    """Apply Gaussian blur with probability p. Safe augmentation for detection."""

    def __init__(self, p=0.3, kernel_sizes=(3, 5)):
        self.p = p
        self.kernel_sizes = kernel_sizes

    def __call__(self, image, target):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            image = TF.gaussian_blur(image, kernel_size=k)
        return image, target


class RandomSizeCrop:
    """Random crop with box filtering — critical for small object detection.

    Crops a random region and keeps boxes whose center falls inside.
    """

    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w, h = image.size
        cw = random.randint(self.min_size, min(w, self.max_size))
        ch = random.randint(self.min_size, min(h, self.max_size))
        region = T.RandomCrop.get_params(image, (ch, cw))
        return _crop(image, target, region)


def _crop(image, target, region):
    """Crop image and adjust target boxes accordingly."""
    top, left, h, w = region
    image = TF.crop(image, top, left, h, w)

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].clone()  # xyxy absolute
        # Shift boxes relative to crop origin
        boxes[:, 0::2] -= left
        boxes[:, 1::2] -= top
        # Clamp to crop boundaries
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Keep boxes whose center is inside the crop and have valid area
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        keep = (cx > 0) & (cx < w) & (cy > 0) & (cy < h) & (bw > 1) & (bh > 1)

        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]

    target["size"] = torch.tensor([h, w])
    return image, target


def make_train_transforms(
    train_sizes,
    max_size,
    use_color_jitter=False,
    color_jitter_kwargs=None,
    use_random_erasing=False,
    random_erasing_p=0.25,
    use_random_crop=False,
    random_crop_min=256,
    random_crop_max=384,
    use_gaussian_blur=False,
    gaussian_blur_p=0.3,
):
    transforms = []
    if use_color_jitter:
        kw = color_jitter_kwargs or {}
        transforms.append(ColorJitter(**kw))
    if use_gaussian_blur:
        transforms.append(RandomGaussianBlur(p=gaussian_blur_p))
    # NOTE: No RandomHorizontalFlip — flipping corrupts digit identity (6↔9, mirrored 2/3/7)
    transforms.append(RandomResize(train_sizes, max_size=max_size))
    if use_random_crop:
        transforms.append(RandomSizeCrop(random_crop_min, random_crop_max))
    transforms += [
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if use_random_erasing:
        transforms.append(RandomErasing(p=random_erasing_p))
    return Compose(transforms)


def make_val_transforms(val_size, max_size):
    return Compose([
        Resize(val_size, max_size=max_size),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DigitDetectionDataset(Dataset):
    """COCO-format dataset for digit detection."""

    def __init__(self, img_dir, json_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(json_file, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.img_ids = [img["id"] for img in data["images"]]

        # Group annotations by image_id
        self.annotations = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])   # xyxy absolute
                labels.append(ann["category_id"])      # 1-indexed

        target = {
            "image_id": torch.tensor([img_id]),
            "boxes": torch.tensor(boxes, dtype=torch.float32)
                    if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
                      if labels else torch.zeros(0, dtype=torch.long),
            "orig_size": torch.tensor([img_info["height"], img_info["width"]]),
            "size": torch.tensor([img_info["height"], img_info["width"]]),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


# ---------------------------------------------------------------------------
# Collate function: pad images to same size within a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images, targets = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)  # True = padding

    new_targets = []
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded[i, :, :h, :w] = img
        masks[i, :h, :w] = False   # real pixels are NOT masked

        # Rescale boxes from valid-image-normalised to padded-tensor-normalised
        # so model predictions (in padded space) match target coordinates.
        tgt = {k: v.clone() if isinstance(v, torch.Tensor) else v
               for k, v in targets[i].items()}
        if "boxes" in tgt and len(tgt["boxes"]) > 0:
            boxes = tgt["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / max_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / max_h)
            tgt["boxes"] = boxes
        new_targets.append(tgt)

    return padded, masks, new_targets


def collate_fn_train(batch):
    """Like collate_fn but places each image at a random position within the
    padded tensor (instead of always top-left).  Forces the model to learn
    digit appearance, not absolute position."""
    import random
    images, targets = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    new_targets = []
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]

        pad_y = random.randint(0, max_h - h) if max_h > h else 0
        pad_x = random.randint(0, max_w - w) if max_w > w else 0

        padded[i, :, pad_y:pad_y + h, pad_x:pad_x + w] = img
        masks[i, pad_y:pad_y + h, pad_x:pad_x + w] = False

        tgt = {k: v.clone() if isinstance(v, torch.Tensor) else v
               for k, v in targets[i].items()}
        if "boxes" in tgt and len(tgt["boxes"]) > 0:
            boxes = tgt["boxes"]
            # Step 1: scale w/h outputs (indices 2,3) from image-normalised
            #         to padded-tensor-normalised (same as original collate_fn)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / max_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / max_h)
            # Step 2: shift cx/cy (indices 0,1) by the random padding offset
            boxes[:, 0] = boxes[:, 0] + pad_x / max_w
            boxes[:, 1] = boxes[:, 1] + pad_y / max_h
            tgt["boxes"] = boxes
        new_targets.append(tgt)

    return padded, masks, new_targets
