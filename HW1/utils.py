import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import v2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, List


def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_num_parameters(model: torch.nn.Module) -> tuple:

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def verify_parameter_constraint(model: torch.nn.Module, max_params: int = 100e6) -> bool:

    total, trainable = get_num_parameters(model)
    within_limit = total < max_params
    
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Parameter Constraint: {'✓ PASS' if within_limit else '✗ FAIL'} (< {max_params/1e6:.0f}M)")
    
    return within_limit


def create_directories(checkpoint_dir: str = "./checkpoints", log_dir: str = "./logs"):

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./models", exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    checkpoint_dir: str = "./checkpoints"
):

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    
    save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0)


def set_seed(seed: int = 42):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


class RandomResizedCrop:

    def __init__(self, size: int = 224, scale: Tuple[float, float] = (0.08, 1.0)):
        self.transform = T.RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=(0.75, 1.33),
            interpolation=T.InterpolationMode.BILINEAR
        )
    
    def __call__(self, img):
        return self.transform(img)


class RandAugment:

    OPERATIONS = [
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
        'AutoContrast', 'Invert', 'Equalize', 'Solarize', 'Posterize',
        'Contrast', 'Color', 'Brightness', 'Sharpness', 'Cutout'
    ]
    
    def __init__(self, n: int = 2, m: int = 9, image_size: int = 224):

        self.n = n
        self.m = m
        self.image_size = image_size
    
    def _apply_operation(self, img: Image.Image, op: str, magnitude: float) -> Image.Image:

        w, h = img.size
        
        if op == 'ShearX':
            magnitude = np.clip(magnitude / 30 * 0.3, 0, 0.3)
            img = img.transform((w, h), Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
        
        elif op == 'ShearY':
            magnitude = np.clip(magnitude / 30 * 0.3, 0, 0.3)
            img = img.transform((w, h), Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
        
        elif op == 'TranslateX':
            magnitude = int(magnitude / 30 * w * 0.3)
            img = img.transform((w, h), Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
        
        elif op == 'TranslateY':
            magnitude = int(magnitude / 30 * h * 0.3)
            img = img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, magnitude))
        
        elif op == 'Rotate':
            magnitude = np.clip(magnitude / 30 * 30, 0, 30)
            img = img.rotate(magnitude)
        
        elif op == 'AutoContrast':
            img = T.functional.autocontrast(img)
        
        elif op == 'Invert':
            img = T.functional.invert(img)
        
        elif op == 'Equalize':
            img = T.functional.equalize(img)
        
        elif op == 'Solarize':
            threshold = int(magnitude / 30 * 256)
            img = T.functional.solarize(img, threshold)
        
        elif op == 'Posterize':
            bits = max(1, int(8 - magnitude / 30 * 4))
            img = T.functional.posterize(img, bits)
        
        elif op == 'Contrast':
            magnitude = 1 + np.clip(magnitude / 30 * 0.9, 0, 0.9)
            img = T.functional.adjust_contrast(img, magnitude)
        
        elif op == 'Color':
            magnitude = 1 + np.clip(magnitude / 30 * 0.9, 0, 0.9)
            img = T.functional.adjust_saturation(img, magnitude)
        
        elif op == 'Brightness':
            magnitude = 1 + np.clip(magnitude / 30 * 0.9, 0, 0.9)
            img = T.functional.adjust_brightness(img, magnitude)
        
        elif op == 'Sharpness':
            magnitude = 1 + np.clip(magnitude / 30 * 0.9, 0, 0.9)
            img = T.functional.adjust_sharpness(img, magnitude)
        
        elif op == 'Cutout':
            size = int(magnitude / 30 * min(w, h) * 0.5)
            x = np.random.randint(0, max(1, w - size))
            y = np.random.randint(0, max(1, h - size))
            img_arr = np.array(img, dtype=np.float32)
            img_arr[y:y+size, x:x+size] = 128
            img = Image.fromarray(np.uint8(img_arr))
        
        return img
    
    def __call__(self, img: Image.Image) -> Image.Image:
        for _ in range(self.n):
            op = np.random.choice(self.OPERATIONS)
            img = self._apply_operation(img, op, self.m)
        return img


class CutMix:

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):

        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple:

        if torch.rand(1).item() > self.prob:
            return images, None, 1.0, None
        
        batch_size = images.size(0)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = torch.tensor(lam, device=images.device, dtype=images.dtype)

        idx = torch.randperm(batch_size, device=images.device)

        _, _, H, W = images.shape

        cut_ratio = np.sqrt(1.0 - lam.item())
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
        lam = torch.tensor(actual_lam, device=images.device, dtype=images.dtype)

        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        y_a = labels
        y_b = labels[idx]
        labels_mixed = (y_a, y_b)
        
        return images, labels_mixed, lam, idx


def apply_cutmix_to_batch(images, labels, cutmix_aug) -> Tuple:
    return cutmix_aug(images, labels)


def get_advanced_transforms(image_size: int = 224):

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        RandAugment(n=2, m=9, image_size=image_size),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1),
        T.RandomRotation(degrees=22),
        T.ToTensor(),
        T.RandomErasing(p=0.35, scale=(0.02, 0.33)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    val_transform = T.Compose([
        T.Resize(int(image_size * 1.15)), 
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transform, val_transform


def get_data_transforms(image_size: int = 224):
    return get_advanced_transforms(image_size)

