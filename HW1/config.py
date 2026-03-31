DATA_ROOT = "./data"
TRAIN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/val"
TEST_DIR = f"{DATA_ROOT}/test"

MODEL_NAME = "ecaresnet50d"  
NUM_CLASSES = 100
PRETRAINED = True
WEIGHTS_NAME = "IMAGENET1K_V1"  
MODEL_CHECKPOINT = "./models/best_model.pth"

BATCH_SIZE = 32 
LEARNING_RATE = 0.008  
MOMENTUM = 0.9 
WEIGHT_DECAY = 3e-4 
NUM_EPOCHS = 180 
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 30  

LR_SCHEDULER = "multistep"  
COSINE_T_MAX = 145 
COSINE_ETA_MIN = 1e-5  
MULTISTEP_MILESTONES = [30, 120] 
MULTISTEP_GAMMA = 0.1  

IMAGE_SIZE = 224  
TRAIN_TRANSFORMS = {
    "resize": 256, 
    "center_crop": True,  
    "random_crop": 224, 
    "random_flip": True,  
    "color_jitter": {  
        "brightness": 0.35,  
        "contrast": 0.35,    
        "saturation": 0.35, 
    },
    "random_rotation": 18, 
    "random_erasing": { 
        "probability": 0.25,
        "scale": (0.02, 0.33),
    },
}

VAL_TEST_TRANSFORMS = {
    "resize": 256,  
    "center_crop": 224,  
}

DEVICE = "cuda"  
NUM_WORKERS = 4
PIN_MEMORY = True

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

OPTIMIZER = "sgd" 
LOSS_FUNCTION = "cross_entropy"
LABEL_SMOOTHING = 0.1  

USE_ADVANCED_AUG = True  
RANDAUGMENT_N = 2 
RANDAUGMENT_M = 9
CUTMIX_ALPHA = 1.0 
CUTMIX_PROB = 0.5 


