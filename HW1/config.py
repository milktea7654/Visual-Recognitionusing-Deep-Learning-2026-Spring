"""
Configuration file for the 100-class image classification task.
"""

# Data paths
DATA_ROOT = "./data"
TRAIN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/val"
TEST_DIR = f"{DATA_ROOT}/test"

# Model configuration
MODEL_NAME = "ecaresnet50d"  # Options: resnet50_inat, resnext101_32x8d, resnet101, ecaresnet50d, gcresnet50t
NUM_CLASSES = 100
PRETRAINED = True
WEIGHTS_NAME = "IMAGENET1K_V1"  # Note: All models now use timm/torchvision with best available ImageNet weights
MODEL_CHECKPOINT = "./models/best_model.pth"

# Training hyperparameters
BATCH_SIZE = 32 # Increased for better gradient estimation
LEARNING_RATE = 0.008  # Back to 0.008 (training data is now easier, can use higher LR)
MOMENTUM = 0.9  # For SGD optimizer
WEIGHT_DECAY = 3e-4  # Reduced from 5e-4 to 3e-4 (lighter regularization for easier training)
NUM_EPOCHS = 180  # Increased from 150 to allow better convergence with new augmentation  # Extended training for convergence
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 30  # Increased from 25 to 30 for more stable training

# Learning rate scheduler
LR_SCHEDULER = "multistep"  # Changed from cosine to multistep for better control
COSINE_T_MAX = 145  # NUM_EPOCHS - WARMUP_EPOCHS (not used for multistep)
COSINE_ETA_MIN = 1e-5  # not used for multistep
# MultiStepLR configuration
MULTISTEP_MILESTONES = [30, 120]  # Epochs where LR drops (at 60 and 120)
MULTISTEP_GAMMA = 0.1  # LR multiplier at each milestone (0.1x = drop to 10%)

# Data augmentation (with standard augmentations for better generalization)
IMAGE_SIZE = 224  # Use 224x224 (standard ImageNet size)
TRAIN_TRANSFORMS = {
    "resize": 512,  # Resize to 256 first
    "center_crop": True,  # Don't center crop
    "random_crop": 368,  # ✓ Random crop to 224x224 for data augmentation
    "random_flip": True,  # ✓ Random horizontal flip (50%)
    "color_jitter": {  # ✓ Enhanced color jittering for robustness
        "brightness": 0.35,  # Increased from 0.25 to 0.35 (stronger augmentation)
        "contrast": 0.35,    # Increased from 0.25 to 0.35
        "saturation": 0.35,  # Increased from 0.25 to 0.35
    },
    "random_rotation": 18,  # Increased from 12 to 18 degrees (stronger)
    "random_erasing": {  # ✓ Random erasing (cutout-style regularization)
        "probability": 0.25,  # Increased from 0.15 to 0.25 (more frequent)
        "scale": (0.02, 0.33),
    },
}

VAL_TEST_TRANSFORMS = {
    "resize": 512,  # Resize to 256
    "center_crop": 368,  # Center crop to 224x224 for validation/test (no augmentation)
}

# Hardware
DEVICE = "cuda"  # Use GPU with PyTorch 2.8.0 + CUDA 12.8
NUM_WORKERS = 4
PIN_MEMORY = True

# Directories
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# Optimization
OPTIMIZER = "sgd"  # SGD+Momentum better for ResNet on real-world images
LOSS_FUNCTION = "cross_entropy"  # Helps with classification confidence
LABEL_SMOOTHING = 0.1  # Reduced from 0.2 to 0.1 (lighter regularization)

# Advanced Augmentations (for animal/plant classification)
USE_ADVANCED_AUG = True  # Enable RandomResizedCrop + RandAugment + CutMix
RANDAUGMENT_N = 2  # Increased from 1 to 2 (stronger augmentation)
RANDAUGMENT_M = 9  # Increased from 5 to 9 (stronger magnitude)
CUTMIX_ALPHA = 1.0  # Increased from 0.5 to 1.0 (stronger mixing)
CUTMIX_PROB = 0.5  # Increased from 0.3 to 0.5 (more frequent)


