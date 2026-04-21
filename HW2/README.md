# NYCU Computer Vision 2026 HW2

- **Student ID:** 112550200
- **Name:** Zheng Wu Qian

## Introduction

This repository contains the implementation for NYCU Computer Vision 2026 HW2. The goal is to perform object detection on a multiple-digit dataset. The core architecture relies on **DINO (DEtection TRansformer with Improved DeNoising)**, utilizing a ResNet-50 backbone. 

## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

### Training

How to train your model. (Default configurations are set in `config.py`)

```bash
python train.py
```

### Inference

How to run inference. (Replace the checkpoint path with your actual best model)

```bash
python inference.py --checkpoint output/20260421_015205/best.pth --batch_size 8
```

## Performance Snapshot

Insert a screenshot of the leaderboard here.

*(Please provide the screenshot of your highest mAP submission below)*
![Leaderboard Snapshot](leaderboard.png)
