import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import argparse
from PIL import Image
from datetime import datetime

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from config import *
from utils import (
    setup_logging,
    get_num_parameters,
    create_directories,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    AverageMeter,
    accuracy,
    get_advanced_transforms,
    CutMix)


class NumericSortedImageFolder(datasets.ImageFolder):
    """ImageFolder with numeric directory sorting.

    Extends ImageFolder to sort directories numerically rather than
    lexicographically, useful for class folders named 0-99.
    """

    def find_classes(self, directory):
        classes = sorted(
            [d.name for d in os.scandir(directory) if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else float('inf')
        )
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images without labels.

    Attributes:
        image_dir: Directory path containing images.
        transform: Optional image transformation pipeline.
        image_paths: List of paths to image files.
    """

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class ResNetClassifier:
    """ResNet-based image classifier wrapper.

    Supports multiple model architectures including ResNet50, ResNeXt-101,
    ECA-ResNet50d, and GCResNet50t with optional pretrained weights.

    Attributes:
        device: GPU/CPU device for model.
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
    """

    def __init__(
            self,
            model_name: str = "ecaresnet50d",
            num_classes: int = 100,
            pretrained: bool = True,
            weights: str = "IMAGENET1K_V1",
            device: str = "cuda"):

        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes

        if model_name in ['resnet50_inat', 'resnext101_32x8d']:
            if model_name == 'resnet50_inat':
                try:
                    self.model = models.resnet50(weights='DEFAULT')
                    msg = " Loading ResNet50 with iNaturalist weights"
                    print(f"{msg}")
                except BaseException:
                    self.model = models.resnet50(weights=None)
                    msg = " Loading ResNet50 (ImageNet-1K fallback)"
                    print(f"{msg}")
            else:
                self.model = models.resnext101_32x8d(weights='DEFAULT')
                msg = " Loading ResNeXt-101 32x8d with ImageNet-1K"
                print(f"{msg}")

            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        elif model_name in ['resnet101', 'ecaresnet50d', 'gcresnet50t']:
            if not TIMM_AVAILABLE:
                msg = "timm required. Install: pip install timm"
                raise ImportError(msg)

            timm_model_map = {
                'resnet101': 'resnet101',
                'ecaresnet50d': 'ecaresnet50d',
                'gcresnet50t': 'gcresnet50t'
            }

            timm_name = timm_model_map.get(model_name, model_name)

            weight_info = {
                'resnet101': 'ImageNet-1K (a1h)',
                'ecaresnet50d': 'RA-Augmented ImageNet (best)',
                'gcresnet50t': 'ImageNet-1K'}

            if pretrained:
                weights_desc = weight_info.get(model_name, 'ImageNet-1K')
                msg = f"✓ Loading {model_name} from timm (pretrained)"
                print(msg)
                print(f"  Weights: {weights_desc}")

            self.model = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=num_classes)
        else:
            valid = "resnet50_inat, resnext101_32x8d, resnet101, ecaresnet50d"
            msg = f"Unsupported model: {model_name}. Use: {valid}, gcresnet50t"
            raise ValueError(msg)

        self.model = self.model.to(device)

        total_params, trainable_params = get_num_parameters(self.model)
        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        ok = '✓ PASS' if total_params < 100e6 else '✗ FAIL'
        print(f"Parameter Limit (< 100M): {ok}")
        print(f"{'='*60}\n")

        if total_params >= 100e6:
            raise ValueError(
                f"Model exceeds 100M parameter limit: {total_params/1e6:.2f}M")

    def get_model(self) -> nn.Module:
        """Return the model."""
        return self.model


def get_data_transforms():
    train_transform_list = []
    resize_size = TRAIN_TRANSFORMS.get("resize", 256)
    train_transform_list.append(
        transforms.Resize((resize_size, resize_size), antialias=True)
    )

    if TRAIN_TRANSFORMS.get("random_crop"):
        crop_size = TRAIN_TRANSFORMS.get("random_crop")
        train_transform_list.append(transforms.RandomCrop(crop_size))
    elif TRAIN_TRANSFORMS.get("center_crop"):
        crop_size = TRAIN_TRANSFORMS.get("center_crop")
        train_transform_list.append(transforms.CenterCrop(crop_size))

    if TRAIN_TRANSFORMS.get("random_flip", False):
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if TRAIN_TRANSFORMS.get("color_jitter"):
        jitter_config = TRAIN_TRANSFORMS["color_jitter"]
        train_transform_list.append(transforms.ColorJitter(
            brightness=jitter_config.get("brightness", 0),
            contrast=jitter_config.get("contrast", 0),
            saturation=jitter_config.get("saturation", 0)
        ))

    if TRAIN_TRANSFORMS.get("random_rotation", 0) > 0:
        train_transform_list.append(
            transforms.RandomRotation(TRAIN_TRANSFORMS["random_rotation"])
        )

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if TRAIN_TRANSFORMS.get("random_erasing"):
        erasing_config = TRAIN_TRANSFORMS["random_erasing"]
        train_transform_list.append(transforms.RandomErasing(
            p=erasing_config.get("probability", 0),
            scale=erasing_config.get("scale", (0.02, 0.33))
        ))

    train_transform = transforms.Compose(train_transform_list)

    val_test_list = []

    resize_size = VAL_TEST_TRANSFORMS.get("resize", 256)
    val_test_list.append(
        transforms.Resize(
            (resize_size,
             resize_size),
            antialias=True))

    if VAL_TEST_TRANSFORMS.get("center_crop"):
        crop_size = VAL_TEST_TRANSFORMS.get("center_crop")
        val_test_list.append(transforms.CenterCrop(crop_size))

    val_test_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose(val_test_list)

    return train_transform, val_test_transform


def get_dataloaders(
        batch_size: int = 64,
        num_workers: int = 4,
        use_advanced_aug: bool = True):

    if use_advanced_aug:
        train_transform, val_test_transform = get_advanced_transforms(
            image_size=224)
        aug_msg = "✓ Advanced: ResizedCrop, RandAugment, ColorJitter"
        print(aug_msg)
    else:
        train_transform, val_test_transform = get_data_transforms()
        print("✓ Using standard augmentations")

    train_dataset = NumericSortedImageFolder(
        TRAIN_DIR, transform=train_transform)
    val_dataset = NumericSortedImageFolder(
        VAL_DIR, transform=val_test_transform)

    test_dataset = SimpleImageDataset(TEST_DIR, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


def get_optimizer(model: nn.Module, learning_rate: float = 0.001):

    if OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )

    return optimizer


def get_scheduler(
        optimizer: optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5):

    if LR_SCHEDULER == "multistep":
        adjusted = [m - warmup_epochs if m > warmup_epochs else m
                    for m in MULTISTEP_MILESTONES]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                current_epoch = epoch - warmup_epochs
                decay_count = sum(
                    1 for m in adjusted if current_epoch >= m)
                return (MULTISTEP_GAMMA ** decay_count)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                pi_term = torch.pi * (epoch - warmup_epochs)
                cos_arg = pi_term / (num_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(cos_arg)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler


def get_loss_function():
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    return loss_fn


def train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch,
        logger):
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    cutmix = CutMix(alpha=1.0, prob=0.5)

    end = time.time()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in pbar:
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, labels_mixed, lam, idx = cutmix(images, labels)

        outputs = model(images)

        if labels_mixed is not None:
            y_a, y_b = labels_mixed
            loss = lam * criterion(outputs, y_a) + \
                (1 - lam) * criterion(outputs, y_b)
        else:
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%'
        })

    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, logger):
    model.eval()

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%'
            })

    logger.info(
        f"Validation Results - Loss: {losses.avg:.4f}, "
        f"Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%"
    )

    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet for 100-class classification')
    parser.add_argument(
        '--model',
        type=str,
        default='ecaresnet50d',
        choices=[
            'resnet50_inat',
            'resnext101_32x8d',
            'resnet101',
            'ecaresnet50d',
            'gcresnet50t'],
        help='Model: resnet50_inat, resnext101_32x8d, resnet101, ecaresnet50d')
    parser.add_argument('--weights', type=str, default='IMAGENET1K_V1',
                        choices=['IMAGENET1K_V1', 'IMAGENET1K_V2', 'SWSL'],
                        help='Pretrained weights (SWSL only for ResNet-101)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS,
        help='Number of epochs')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size')
    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint')
    parser.add_argument(
        '--device',
        type=str,
        default=DEVICE,
        choices=[
            'cuda',
            'cpu'])

    args = parser.parse_args()

    if args.device == 'cuda':
        try:
            test_tensor = torch.randn(2, 64, 56, 56).to('cuda')
            bn = torch.nn.BatchNorm2d(64).to('cuda')
            bn(test_tensor)
            torch.cuda.synchronize()
            print("✓ CUDA device ready for training")
        except RuntimeError:
            print(" CUDA incompatibility detected")
            print(
                "→ Switching to CPU training (compatible with PyTorch)")
            args.device = 'cpu'

    set_seed(42)
    create_directories()
    logger = setup_logging()
    logger.info("\n" + "=" * 70)
    model_info = {
        'resnet101': 'ResNet101 (Baseline)',
        'eca_resnet50': 'ECA-ResNet50 (Efficient Channel Attention)',
        'gc_resnet50': 'GC-ResNet50 (Global Context)'
    }
    logger.info(f"MODEL: {model_info.get(args.model, args.model)}")
    logger.info("=" * 70)

    classifier = ResNetClassifier(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        weights=args.weights,
        device=args.device
    )
    model = classifier.get_model()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Model: {model_info.get(args.model, args.model)}")
    if args.model == 'resnet101':
        logger.info(f"Pretrained: True ({args.weights} weights)")
    else:
        logger.info(f"Pretrained: True (ImageNet-1K weights)")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    logger.info(f"Device: {args.device.upper()}")

    logger.info("\nTraining Parameters:")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Learning Rate: {args.lr}")
    opt_msg = f"{OPTIMIZER.upper()} (m={MOMENTUM}, decay={WEIGHT_DECAY})"
    logger.info(f"  - Optimizer: {opt_msg}")
    sch_msg = f"{LR_SCHEDULER.upper()} (T_max={COSINE_T_MAX}, "
    sch_msg += f"min={COSINE_ETA_MIN})"
    logger.info(f"  - Scheduler: {sch_msg}")
    logger.info(f"  - Warmup Epochs: {WARMUP_EPOCHS}")
    logger.info(f"  - Max Epochs: {args.epochs}")
    loss_msg = f"{LOSS_FUNCTION.upper()} (label_smoothing={LABEL_SMOOTHING})"
    logger.info(f"  - Loss Function: {loss_msg}")
    logger.info(f"  - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    logger.info("Data Augmentation:")
    logger.info(f"  - Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info(
        f"  - Random Flip: {TRAIN_TRANSFORMS.get('random_flip', False)}")
    logger.info(
        f"  - Color Jitter: {TRAIN_TRANSFORMS.get('color_jitter', None)}")
    logger.info(
        f"  - Random Rotation: {TRAIN_TRANSFORMS.get('random_rotation', 0)}°")
    logger.info(
        f"  - Random Erasing: {TRAIN_TRANSFORMS.get('random_erasing', None)}")
    logger.info("=" * 70)

    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        use_advanced_aug=True
    )

    logger.info("Dataset:")
    n_train = len(train_loader.dataset)
    n_batch = len(train_loader)
    logger.info(f"  - Train: {n_train} images ({n_batch} batches)")
    val_msg = f"{len(val_loader.dataset)} images ({len(val_loader)} batches)"
    logger.info(f"  - Val: {val_msg}")
    n_test = len(test_loader.dataset)
    n_batch = len(test_loader)
    logger.info(f"  - Test: {n_test} images ({n_batch} batches)\n")

    optimizer = get_optimizer(model, learning_rate=args.lr)
    scheduler = get_scheduler(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=WARMUP_EPOCHS)
    criterion = get_loss_function()

    start_epoch = 0
    best_acc = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer)

    logger.info(f"Starting training for {args.epochs} epochs...")
    no_improve_count = 0

    run_name = f"train_{int(datetime.now().timestamp())}"
    run_dir = os.path.join("./runs", run_name)
    writer = SummaryWriter(log_dir=run_dir)
    logger.info(f"TensorBoard logs will be saved to: {run_dir}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        tr_loss, tr_acc1, tr_acc5 = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            epoch + 1, logger)

        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, args.device, logger)

        scheduler.step()

        writer.add_scalar('Loss/train', tr_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', tr_acc1, epoch + 1)
        writer.add_scalar('Accuracy/val', val_acc1, epoch + 1)
        writer.add_scalar('Accuracy@5/train', tr_acc5, epoch + 1)
        writer.add_scalar('Accuracy@5/val', val_acc5, epoch + 1)
        writer.add_scalar(
            'LearningRate',
            optimizer.param_groups[0]['lr'],
            epoch + 1)
        writer.add_scalar('epoch', epoch + 1, epoch + 1)

        if val_acc1 > best_acc:
            best_acc = val_acc1
            save_path = save_checkpoint(model, optimizer, epoch + 1, best_acc)
            msg = f"New best model at {save_path} with Acc@1: {best_acc:.2f}%"
            logger.info(msg)
            no_improve_count = 0

            torch.save(model.state_dict(), MODEL_CHECKPOINT)
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            es_msg = f"Early stop: {EARLY_STOPPING_PATIENCE} epochs no improve"
            logger.info(es_msg)
            break

    writer.close()
    logger.info(f"TensorBoard logs saved to: {run_dir}")
    logger.info(
        f"\nTraining completed. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
