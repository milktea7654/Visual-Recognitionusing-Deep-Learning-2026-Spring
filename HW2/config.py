"""Configuration for DETR digit detection (HW2)."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "data"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "valid")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VAL_JSON = os.path.join(DATA_ROOT, "valid.json")

OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
NUM_CLASSES = 10        # digits 0-9 (category_id 1-10 in COCO JSON)
# Images have max 6 digits; 20 queries gives safe headroom with less noise
NUM_QUERIES = 15
HIDDEN_DIM = 128        # reduced for simple digit task
NHEADS = 8              # number of attention heads (must divide HIDDEN_DIM)
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512   # 4× HIDDEN_DIM
DROPOUT = 0.2
PRETRAINED_BACKBONE = True

# Deformable attention specific
N_FEATURE_LEVELS = 3    # C3 / C4 / C5 from ResNet-50
N_POINTS = 4            # sampling points per attention head per level
USE_CHECKPOINT = False  # off for speed; BS=2 fits VRAM with C2 features

# Iterative bounding box refinement (each decoder layer refines bbox)
ITERATIVE_REFINE = True
# Auxiliary losses from intermediate decoder layers (needed for from-scratch convergence)
AUX_LOSS = True

# Backbone freeze option:
#   True  = only train layers 3/4 of ResNet (freeze layer1/2)
#   False = train all backbone layers
FREEZE_BACKBONE_EARLY = False

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Images are small (mean 128x57 px), so a larger batch fits in memory
BATCH_SIZE = 4       # reduced for C2 feature VRAM
GRAD_ACCUM_STEPS = 4    # accumulate 8 steps → effective batch = 16
# Deformable DETR converges ~10x faster than vanilla DETR; 100 epochs sufficient for 30k dataset
NUM_EPOCHS = 50

# Optimizer
OPTIMIZER = "adamw"     # "adamw" | "adam" | "sgd"
LR = 1e-4               # from-scratch training needs higher LR
LR_BACKBONE = 1e-5      # backbone is ImageNet-pretrained so keep 10x lower
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9          # only used when OPTIMIZER = "sgd"

# LR scheduler
# "step"      : multiply LR by LR_GAMMA every LR_DROP epochs
# "cosine"    : cosine annealing over NUM_EPOCHS
# "multistep" : decay at epochs listed in LR_MILESTONES
LR_SCHEDULER = "multistep"
LR_DROP = 80
LR_GAMMA = 0.1
LR_MILESTONES = [20, 35]

GRAD_CLIP = 0.1

# Mixed-precision training (requires CUDA)
AMP = True              # True = use AMP for faster training
AMP_DTYPE = "bfloat16"  # bf16 is faster than fp16 on Ampere+ GPUs (RTX 30xx/40xx/50xx)

# Resume from checkpoint (set to path string to resume, None to start fresh)
RESUME = None          # fresh start — new architecture dims incompatible with old checkpoint

# ---------------------------------------------------------------------------
# Loss coefficients (Hungarian matching + training loss)
# ---------------------------------------------------------------------------
# Matching costs — focal-loss-based class cost for better convergence
COST_CLASS = 2.0        # matching: class cost weight (focal-based)
COST_BBOX = 5.0         # matching: L1 bbox cost weight
COST_GIOU = 2.0         # matching: GIoU cost weight

# Training loss weights (standard Deformable DETR weights)
LOSS_CLS = 2.0          # sigmoid focal loss weight
LOSS_BBOX = 5.0         # L1 bbox loss weight (standard Deformable DETR)
LOSS_GIOU = 2.0         # GIoU weight

# Focal loss hyperparameters (replaces softmax CE + eos_coef)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ---------------------------------------------------------------------------
# Data augmentation / resize
# ---------------------------------------------------------------------------
# Images are small (mean 128x57 px, max 876 px wide).
# C3 (stride 8) in N_FEATURE_LEVELS=3 already handles fine detail.
# Reduce scale range to limit interpolation artifacts on tiny digits.
TRAIN_SIZES = [192, 224, 256]
VAL_SIZE = 224
MAX_SIZE = 448

# Extra augmentations (ColorJitter applied before resize)
# House number images vary widely in lighting and color — keep jitter on
USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.05

# Random crop — DISABLED for digit detection: images are tiny (128×57 px avg),
# cropping risks cutting digits in half and corrupting labels.
USE_RANDOM_CROP = False
RANDOM_CROP_MIN = 256
RANDOM_CROP_MAX = 384

# Random erasing (applied after normalization)
USE_RANDOM_ERASING = True
RANDOM_ERASING_P = 0.2

# Gaussian blur (safe augmentation, simulates out-of-focus cameras)
USE_GAUSSIAN_BLUR = False
GAUSSIAN_BLUR_P = 0.3

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Lower threshold helps recall small digits; raise if too many false positives
SCORE_THRESHOLD = 0
# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEVICE = "cuda"
NUM_WORKERS = 8         # 8 workers; GPU step=400ms so data loading must stay ahead
SEED = 42

