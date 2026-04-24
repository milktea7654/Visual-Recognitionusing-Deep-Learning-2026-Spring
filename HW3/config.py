"""config.py – All hyper-parameters for HW3 Mask2Former training."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = "data/train"
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

NUM_CLASSES = 4        # class1 … class4 (background handled by model)
MAX_SIZE = 768         # cap long edge for small images only; large images are CROPPED
MIN_SIZE = 256         # minimum output dim; small images padded to this
CROP_SIZE = 768        # random-crop target for large images (train) / sliding-window tile size (inference)
SIZE_DIVISIBILITY = 32 # ConvNeXt downsamples ×32; all dims must be multiple
SLIDING_WINDOW_OVERLAP = 0.2  # fractional overlap between tiles during inference
MAX_INSTANCES = 100    # cap instances per image; 200 caused OOM in loss padding

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

PRETRAINED = True      # load ImageNet-1K backbone ONLY (no COCO weights)
NUM_QUERIES = 200      # object queries per image

# --- Backbone (ConvNeXt-V2) ---
# Options tested: "convnextv2-base", "convnextv2-large", "convnext-base"
# Model IDs must exist on HuggingFace under the "facebook/" namespace.
BACKBONE = "convnextv2-base-1k-224"        # HF model id suffix (facebook/{BACKBONE})
BACKBONE_DEPTHS       = [3, 3, 27, 3]       # stage depths for base variant
BACKBONE_HIDDEN_SIZES = [128, 256, 512, 1024]  # channel widths per stage
BACKBONE_DROP_PATH    = 0.3                  # stochastic depth rate

# --- Mask2Former decoder ---
DECODER_FEATURE_SIZE      = 256   # pixel-decoder output channels
DECODER_MASK_FEATURE_SIZE = 256
DECODER_HIDDEN_DIM        = 256
DECODER_ENCODER_LAYERS    = 6
DECODER_DECODER_LAYERS    = 6
DECODER_ATTENTION_HEADS   = 8
DECODER_DIM_FEEDFORWARD   = 2048
DECODER_DROPOUT           = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

SEED = 42
EPOCHS = 300
BATCH_SIZE = 2         # ConvNeXt-Base activations are larger than Swin; 4 OOMs
NUM_WORKERS = 8
USE_AMP = True         # fp16 mixed precision (≈50% VRAM reduction)

WARMUP_EPOCHS = 0      # no warmup (backbone un-frozen from start)

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

OPTIMIZER = "AdamW"
LR = 2e-4              # decoder / head learning rate
LR_BACKBONE = 2e-5     # backbone learning rate (10× lower for fine-tuning)
LR_MIN = 1e-6          # cosine floor
WEIGHT_DECAY = 0.05

GRAD_CLIP = 5.0        # CNNs are more stable; larger clip lets gradients flow

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

SCHEDULER = "OneCycleLR"  # 1-cycle policy: warmup → anneal

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VAL_SCORE_THRESHOLD = 0.05  # lower threshold (0.05) to observe early AP50 > 0


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

SAVE_EVERY = 40         # save checkpoint every N epochs

# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

EARLY_STOP_PATIENCE = 50  # stop if AP50 does not improve for this many epochs
