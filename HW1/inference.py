"""Inference script for generating predictions on test dataset.

This module loads a trained model and generates predictions on the test set,
saving results to a CSV file in the standard Kaggle format.

Typical usage example:
  python inference.py --model ecaresnet50d --checkpoint models/best_model.pth
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from config import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_SIZE,
    MODEL_CHECKPOINT,
    NUM_CLASSES,
    NUM_WORKERS,
    TEST_DIR,
    WEIGHTS_NAME,
)


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

        return image, Path(img_path).stem


def load_model(
    model_name: str = "ecaresnet50d",
    num_classes: int = 100,
    checkpoint_path: str | None = None,
    device: str = "cuda"
) -> nn.Module:
    """Load and initialize a classification model.

    Args:
        model_name: Name of the model architecture to load.
        num_classes: Number of output classes.
        checkpoint_path: Path to model checkpoint file. If None, uses
            MODEL_CHECKPOINT from config.
        device: Device to place model on ('cuda' or 'cpu').

    Returns:
        Loaded model in evaluation mode.

    Raises:
        ImportError: If timm is required but not installed.
        FileNotFoundError: If checkpoint file is not found.
        ValueError: If model_name is not supported.
    """

    if model_name in ['ecaresnet50d', 'gcresnet50t', 'resnet101']:
        if model_name in ['ecaresnet50d', 'gcresnet50t']:
            if not TIMM_AVAILABLE:
                raise ImportError(
                    "timm required. Install: pip install timm")
            print(f"Loading timm model: {model_name}")
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes)
        else:
            print(f"Loading torchvision model: {model_name}")
            model = models.resnet101(weights=WEIGHTS_NAME)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
    elif model_name in ['resnet50_inat', 'resnext101_32x8d']:
        print(f"Loading torchvision model: {model_name}")
        if model_name == 'resnet50_inat':
            model = models.resnet50(weights=None)
        else:
            model = models.resnext101_32x8d(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. See docs for options")

    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINT

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        try:
            model.load_state_dict(state_dict)
            print(f"Model loaded from {checkpoint_path}")
        except RuntimeError as e:
            if "se.fc" in str(e) or "_se_block_cache" in str(e):
                print("Detected SE Block params, adding SE blocks...")
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from train import add_se_blocks_to_resnet
                model = add_se_blocks_to_resnet(model, reduction=16)
                model.load_state_dict(state_dict)
                print(f"Model with SE blocks loaded from {checkpoint_path}")
            else:
                raise e
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model


def get_test_transform() -> transforms.Compose:
    """Get image transformations for test dataset.

    Returns:
        Composed transformation pipeline for test images.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def generate_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    output_path: str = "prediction.csv"
) -> str:
    """Generate predictions on test set and save to CSV.

    Args:
        model: Trained model in evaluation mode.
        test_loader: DataLoader for test dataset.
        device: Device to run model on.
        output_path: Path to save prediction CSV.

    Returns:
        Path to the saved prediction file.
    """

    predictions = []
    image_ids = []

    with torch.no_grad():
        for batch_idx, (images, img_ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            image_ids.extend(img_ids)
            predictions.extend(predicted.cpu().numpy().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Processed {(batch_idx + 1) * test_loader.batch_size}/" +
                    f"{len(test_loader.dataset)} images")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])

        for img_id, pred in zip(image_ids, predictions):
            writer.writerow([img_id, int(pred)])

    print(f"\nPredictions saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions on test set')
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
        help='Model: resnet50_inat, resnext101_32x8d, resnet101, etc.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='prediction.csv',
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=DEVICE,
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("Creating test loader...")
    test_transform = get_test_transform()
    test_dataset = SimpleImageDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")

    print("Generating predictions...")
    generate_predictions(model, test_loader, args.device, args.output)


if __name__ == "__main__":
    main()
