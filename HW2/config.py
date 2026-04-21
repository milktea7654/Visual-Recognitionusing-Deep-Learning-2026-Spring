"""Configuration for DINO digit detection (HW2)."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "data"
TRAIN_ANN = os.path.join(DATA_ROOT, "train.json")
VAL_ANN = os.path.join(DATA_ROOT, "valid.json")

OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
BACKBONE = "resnet50"
DILATION = False
RETURN_INTERM_INDICES = [1, 2, 3]

# Positional encoding (separate H / W temperatures for rectangular images)
PE_TEMPERATURE_H = 20
PE_TEMPERATURE_W = 20

# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
NUM_CLASSES = 11  # digits 0-9 + background class
NUM_QUERIES = 30  # safe headroom for typical ~6 digits/image
HIDDEN_DIM = 256
NHEADS = 4  # each head = 64 dim
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.2
ACTIVATION = "relu"
PRE_NORM = False

# Multi-scale deformable attention
NUM_FEATURE_LEVELS = 4
ENC_POINTS = 4
DEC_POINTS = 4

# Two-stage proposal (encoder proposes top-K, decoder refines)
TWO_STAGE_TYPE = "standard"
TWO_STAGE_LEARN_WH = False
EMBED_INIT_TGT = True

# Decoder sub-layer order
DECODER_SA_TYPE = "sa"
DECODER_MODULE_SEQ = ["sa", "ca", "ffn"]

# Head weight sharing
DEC_CLASS_SHARE = True
DEC_BBOX_SHARE = True
TWO_STAGE_CLASS_SHARE = False
TWO_STAGE_BBOX_SHARE = False

# ---------------------------------------------------------------------------
# Denoising (CDN)
# ---------------------------------------------------------------------------
USE_DN = True
DN_NUMBER = 10
DN_LABEL_NOISE_RATIO = 0.5
DN_BOX_NOISE_SCALE = 0.4
DN_LABELBOOK_SIZE = 11

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
AUX_LOSS = True

# Matching cost weights
COST_CLASS = 2.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# Training loss weights
LOSS_CE_COEF = 1.0
LOSS_BBOX_COEF = 5.0
LOSS_GIOU_COEF = 2.0

FOCAL_ALPHA = 0.2

INTERM_LOSS_COEF = 1.0
NO_INTERM_BOX_LOSS = False

# Post-processing
NUM_SELECT = 30

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
LR_BACKBONE = 1e-5
LR_LINEAR_PROJ_MULT = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.1
LR_DROP = 15
LR_ETA_MIN = 1e-6

RESUME = ""

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# Image Sizes
TRAIN_SHORT_EDGE = 320
TRAIN_MAX_SIZE = 640
VAL_SHORT_EDGE = 320
VAL_MAX_SIZE = 640

# Augmentation: Geometric
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.0
AUG_CROP_PROB = 0.5
AUG_CROP_SCALE = (0.6, 1.0)
AUG_ROTATION_PROB = 0.0
AUG_ROTATION_DEGREES = 5.0
AUG_TRANSLATE_XY = (0.1, 0.1)
AUG_PERSPECTIVE_PROB = 0.5
AUG_PERSPECTIVE_SCALE = 0.1
AUG_MIN_VISIBILITY = 0.5

# Augmentation: Photometric
AUG_COLOR_JITTER_PROB = 0.8
AUG_BLUR_PROB = 0.5
AUG_BLUR_SIGMA = (0.1, 2.0)
AUG_USE_ISO_NOISE = True
AUG_ISO_NOISE_PROB = 0.2
AUG_ISO_NOISE_INTENS = 0.05

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 6


def get_config():
    """Return all settings as a flat dictionary usable by build_model() etc."""
    return {
        # paths
        "data_path": DATA_ROOT,
        "train_json": os.path.basename(TRAIN_ANN),
        "valid_json": os.path.basename(VAL_ANN),
        "output_dir": OUTPUT_DIR,
        # backbone
        "backbone": BACKBONE,
        "dilation": DILATION,
        "return_interm_indices": list(RETURN_INTERM_INDICES),
        "lr_backbone": LR_BACKBONE,
        "pe_temperatureH": PE_TEMPERATURE_H,
        "pe_temperatureW": PE_TEMPERATURE_W,
        # transformer
        "num_classes": NUM_CLASSES,
        "num_queries": NUM_QUERIES,
        "hidden_dim": HIDDEN_DIM,
        "nheads": NHEADS,
        "enc_layers": NUM_ENCODER_LAYERS,
        "dec_layers": NUM_DECODER_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "transformer_activation": ACTIVATION,
        "pre_norm": PRE_NORM,
        "num_feature_levels": NUM_FEATURE_LEVELS,
        "enc_n_points": ENC_POINTS,
        "dec_n_points": DEC_POINTS,
        "two_stage_type": TWO_STAGE_TYPE,
        "two_stage_learn_wh": TWO_STAGE_LEARN_WH,
        "embed_init_tgt": EMBED_INIT_TGT,
        "decoder_sa_type": DECODER_SA_TYPE,
        "decoder_module_seq": list(DECODER_MODULE_SEQ),
        "dec_pred_class_embed_share": DEC_CLASS_SHARE,
        "dec_pred_bbox_embed_share": DEC_BBOX_SHARE,
        "two_stage_class_embed_share": TWO_STAGE_CLASS_SHARE,
        "two_stage_bbox_embed_share": TWO_STAGE_BBOX_SHARE,
        # denoising
        "use_dn": USE_DN,
        "dn_number": DN_NUMBER,
        "dn_label_noise_ratio": DN_LABEL_NOISE_RATIO,
        "dn_box_noise_scale": DN_BOX_NOISE_SCALE,
        "dn_labelbook_size": DN_LABELBOOK_SIZE,
        # loss
        "aux_loss": AUX_LOSS,
        "set_cost_class": COST_CLASS,
        "set_cost_bbox": COST_BBOX,
        "set_cost_giou": COST_GIOU,
        "cls_loss_coef": LOSS_CE_COEF,
        "bbox_loss_coef": LOSS_BBOX_COEF,
        "giou_loss_coef": LOSS_GIOU_COEF,
        "focal_alpha": FOCAL_ALPHA,
        "interm_loss_coef": INTERM_LOSS_COEF,
        "no_interm_box_loss": NO_INTERM_BOX_LOSS,
        "num_select": NUM_SELECT,
        # training
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "lr_linear_proj_mult": LR_LINEAR_PROJ_MULT,
        "weight_decay": WEIGHT_DECAY,
        "clip_max_norm": GRAD_CLIP,
        "lr_drop": LR_DROP,
        "lr_eta_min": LR_ETA_MIN,
        "resume": RESUME,
        # augmentation
        "train_short_edge": TRAIN_SHORT_EDGE,
        "train_max_size": TRAIN_MAX_SIZE,
        "val_short_edge": VAL_SHORT_EDGE,
        "val_max_size": VAL_MAX_SIZE,
        "aug_scale_min": AUG_SCALE_MIN,
        "aug_scale_max": AUG_SCALE_MAX,
        "aug_crop_prob": AUG_CROP_PROB,
        "aug_crop_scale": list(AUG_CROP_SCALE),
        "aug_rotation_prob": AUG_ROTATION_PROB,
        "aug_rotation_degrees": AUG_ROTATION_DEGREES,
        "aug_translate_xy": list(AUG_TRANSLATE_XY),
        "aug_perspective_prob": AUG_PERSPECTIVE_PROB,
        "aug_perspective_scale": AUG_PERSPECTIVE_SCALE,
        "aug_min_visibility": AUG_MIN_VISIBILITY,
        "aug_color_jitter_prob": AUG_COLOR_JITTER_PROB,
        "aug_blur_prob": AUG_BLUR_PROB,
        "aug_blur_sigma": list(AUG_BLUR_SIGMA),
        "aug_use_iso_noise": AUG_USE_ISO_NOISE,
        "aug_iso_noise_prob": AUG_ISO_NOISE_PROB,
        "aug_iso_noise_intensity": AUG_ISO_NOISE_INTENS,
        # misc
        "device": DEVICE,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
    }
