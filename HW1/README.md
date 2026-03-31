# NYCU Computer Vision 2026 HW1 - 100-Class Image Classification

**Student ID:** 112550200
**Name:** Zheng WU QIAN

---

## Introduction

This project implements a deep learning-based image classification model for a 100-class dataset using PyTorch. The model achieves robust performance through carefully tuned hyperparameters, advanced data augmentation, and optimized training strategies.

### Key Features:
- **Multiple Architecture Support:** ECA-ResNet50d, ResNeXt-101, ResNet101, and more
- **Advanced Data Augmentation:** ColorJitter, RandAugment, CutMix, and RandomErasing
- **Learning Rate Scheduling:** MultiStepLR with warmup for controlled convergence
- **TensorBoard Integration:** Real-time monitoring of training metrics
- **Efficient Training:** SGD optimizer with momentum, label smoothing regularization

### Dataset:
- **Total Images:** 23,068
- **Train Set:** 20,724 images (100 classes)
- **Validation Set:** 300 images (3 per class)
- **Test Set:** 2,344 images
- **Image Size:** 224×224 pixels

---

## Environment Setup

### Prerequisites:
- Python 3.9+
- CUDA 12.8 (for GPU acceleration)
- cuDNN support for PyTorch

### Installation:

```bash
# Clone the repository
git clone https://github.com/milktea7654/Visual-Recognitionusing-Deep-Learning-2026-Spring
cd HW1

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies:
```
torch==2.8.0
torchvision==0.18.0
timm==0.9.16
numpy==1.24.3
tensorboard==2.14.1
scikit-learn==1.3.2
pandas==2.0.3
matplotlib==3.7.2
```

---

## Usage

### Training

#### Basic Training (Default Model: ecaresnet50d)
```bash
python train.py
```

#### Train with Specific Model
```bash
# ResNeXt-101 32x8d
python train.py --model resnext101_32x8d

# ResNet-101 from timm
python train.py --model resnet101

# ECA-ResNet50d (recommended)
python train.py --model ecaresnet50d
```

#### Custom Training Arguments
```bash
python train.py \
    --model resnext101_32x8d \
    --epochs 150 \
    --batch-size 32 \
    --lr 0.008 \
    --device cuda
```

### Inference (Generate Predictions)

```bash
# Run inference on test set with best model
python inference.py

# Use specific model checkpoint
python inference.py --model resnext101_32x8d

# Output: prediction.csv with test predictions
```

### Analysis

#### Analyze Validation Errors
```bash
python analyze_predictions.py
```
Generates:
- `validation_errors.csv` - Misclassified samples and predictions
- `class_accuracy.csv` - Per-class accuracy metrics

### Monitor Training with TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

---

## Configuration

All training parameters are centralized in `config.py`:

### Key Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| **BATCH_SIZE** | 32 | Mini-batch size |
| **LEARNING_RATE** | 0.008 | Initial learning rate |
| **NUM_EPOCHS** | 180 | Maximum training epochs |
| **WEIGHT_DECAY** | 3e-4 | L2 regularization |
| **LABEL_SMOOTHING** | 0.1 | Label smoothing strength |

### Learning Rate Scheduler
- **Type:** MultiStepLR (multi-stage decay)
- **Milestones:** [30, 120] epochs
- **Gamma:** 0.1 (decay to 10% at each milestone)
- **Warmup:** 5 epochs with linear increase

### Data Augmentation
| Augmentation | Probability/Strength |
|-------------|---------------------|
| ColorJitter | brightness/contrast/saturation=0.35 |
| Random Rotation | ±18° |
| Random Erasing | probability=0.25 |
| RandAugment | N=2, M=9 |
| CutMix | alpha=1.0, probability=0.5 |

---

## Performance Snapshot

### Validation Metrics (Latest Training Run)
- **Validation Accuracy:** ~92%
- **Top-5 Accuracy:** ~97%
- **Best Epoch:** 61

### Class-wise Performance
**Highest Accuracy Classes:** 
- Classes with 100% accuracy: Multiple
- Avg accuracy: ~88.67%

**Challenging Classes (Lowest Accuracy):**
- Class 11: ~50% accuracy
- Class 20: ~40% accuracy
- Class 44: ~35% accuracy
- Class 58: ~45% accuracy
- Class 72: ~42% accuracy

### Most Common Misclassifications
- Class 20 → Class 26: 4 instances (most frequent confusion)
- Class 10 → Class 15: 2 instances
- Class 5 → Class 8: 2 instances

### Training Time
- **Estimated Duration:** 5.5 hours (RTX 5070 Ti)
- **Memory Usage:** ~14GB VRAM

### Model Size
- **ResNeXt-101 32x8d:** 88M parameters

### Test Set Performance
- **Latest Submission Score:** 0.96

---

## Project Structure

```
HW1/
├── train.py                 # Main training script
├── inference.py             # Test set inference
├── config.py               # Centralized configuration
├── utils.py                # Utility functions
├── README.md               # This file
├── data/
│   ├── train/              # Training images (20,724)
│   ├── val/                # Validation images (300)
│   └── test/               # Test images (2,344)
├── checkpoints/            # Saved model checkpoints
├── models/
│   └── best_model.pth      # Best validation checkpoint
├── logs/                   # Training logs
├── runs/                   # TensorBoard logs
└── prediction.csv          # Test predictions (generated)
```

---

## Key Implementation Details

### Data Pipeline
1. **Resize:** 256×256
2. **Random Crop:** 224×224 for training, Center Crop for validation/test
3. **Augmentations:** Applied in specific order (PIL → Tensor → Normalize)
4. **Normalization:** ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Loss Function
- **Type:** CrossEntropyLoss
- **Label Smoothing:** 0.1
- **Class Weighting:** Disabled (uniform weights)

### Optimizer
- **Type:** SGD with momentum
- **Momentum:** 0.9
- **Weight Decay:** 3e-4 (L2 regularization)

---

## Troubleshooting

### CUDA Out of Memory
- Reduce BATCH_SIZE in config.py
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use smaller model (ecaresnet50d instead of resnext101)

### Learning Rate Scheduler Optimization
**Issue:** Initially used cosine annealing scheduler, but observed poor loss reduction in later training epochs.

**Solution:** Switched to **MultiStepLR** scheduler with milestones at [30, 120] epochs. This multi-stage decay approach provided more stable convergence and better loss decrease in the final training phase.

### Validation Accuracy Plateau
**Issue:** Validation accuracy plateaued around 92% and couldn't improve further.

**Root Cause Analysis:** Dataset analysis revealed that the challenging classes consist primarily of:
- **Fine-grained bird species:** Subtle differences in feather patterns, coloration, and size
- **Plant details:** Minute variations in leaf patterns, flower structures, and plant morphology

These features require higher discriminative capacity to distinguish effectively.

**Attempted Solutions:**
1. **Attention-based models:** Tested squeeze-and-excitation networks and channel attention mechanisms
   - Result: Minimal improvement (~0.5% accuracy gain)
   
2. **iNaturalist pre-training:** Experimented with models pre-trained on iNaturalist dataset (domain-specific)
   - Result: Limited effectiveness on this specific dataset

**Final Solution:** Switched to **larger capacity models** (ECA-ResNet50d, ResNeXt-101) with more parameters to better capture fine-grained visual features. The increased model capacity enabled better learning of subtle discriminative patterns.

**Challenging Classes (Analysis):**
- Class 11 (Birds): ~50% accuracy - Fine feather distinctions
- Class 20 & 26 (Plants): ~40% accuracy - Highly similar leaf structures
- Class 44, 58, 72 (Mixed): ~35-45% accuracy - Subtle morphological differences


---

## Citation & References

### Model Architectures
- **ECA-ResNet:** ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (Wang et al., 2020) - https://arxiv.org/abs/1910.03151
- **Squeeze-and-Excitation Networks:** Hu et al., 2018 - https://arxiv.org/abs/1709.01507
- **CBAM: Convolutional Block Attention Module:** Woo et al., 2018 - https://arxiv.org/abs/1807.06521
- **BAM: Bottleneck Attention Module:** Park et al., 2018 - https://arxiv.org/abs/1807.06674
- **ResNet:** Deep Residual Learning for Image Recognition (He et al., 2015)
- **ResNeXt:** Aggregated Residual Transformations for Deep Neural Networks (Xie et al., 2017)

### Data Augmentation
- **RandAugment:** RandAugment: Practical automated data augmentation with a reduced search space (Cubuk et al., 2019)
- **CutMix:** CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019) - https://arxiv.org/abs/1905.04412
- **AutoAugment:** AutoAugment: Learning Augmentation Policies from Data (Cubuk et al., 2019)
- **Mixup:** mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)

### Fine-grained Image Recognition
- **Fine-Grained Visual Categorization of Aircraft:** Fine-Grained Visual Categorization of Aircraft (Maji et al., 2013)
- **The iNaturalist Species Classification and Detection Dataset:** Horn et al., 2018
- **Attention-based Deep Multiple Instance Learning:** Ilse et al., 2018

### Related Techniques
- **Label Smoothing:** When Does Label Smoothing Help? (Müller et al., 2019)
- **Warmup Strategies:** A Closer Look at Deep Learning Heuristics: Learning rate restart Warmup and Decay (Gotmare et al., 2018)

---


**Last Updated:** 2026-03-31  
**Author:** Zheng Wu Qian
**Student ID:** 112550200
