import yaml
from yacs.config import CfgNode as CN
import torch


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# Batch size for a single GPU, could be 
_C.BATCH_SIZE = 5
# Path to dataset
_C.DATA_PATH = 'C:\\Users\\marouane.tliba\\MedicalImaging\\swinCXR'
# Dataset name
_C.DATASET = 'nih'
# Validation percentage
_C.VAL_SPLIT = 0.10
# Input image size
_C.IMG_SIZE = 224
_C.NUM_MLP_HEADS = 3
_C.NUM_SAMPLES = None

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.PIN_MEMORY = False
# Number of data loading threads
_C.NUM_WORKERS = 1


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# Model type
_C.TYPE = 'swin'
# Model name
_C.NAME = 'boost_swin' #'swin_large_patch4_window7_224_22k-4-batch64'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
_C.RESUME = 'swin_large_patch4_window7_224_22k.pth'
# Number of classes, overwritten in data preparation
_C.NUM_CLASSES = 14
# Dropout rate
_C.DROP_RATE = 0.0
# Drop path rate
_C.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.LABEL_SMOOTHING = 0

# Swin Transformer parameters
_C.PATCH_SIZE = 4
_C.IN_CHANS = 3
_C.EMBED_DIM = 96 # 192
_C.DEPTHS =[2, 2, 6, 2] #  [2, 2, 18, 2]  
_C.NUM_HEADS = [3, 6, 12, 24] # [6, 12, 24, 48] # 
_C.WINDOW_SIZE = 7
_C.MLP_RATIO = 4.
_C.QKV_BIAS = True
_C.QK_SCALE = None
_C.APE = False
_C.PATCH_NORM = True

# Barlow Twins
_C.LAMBDA = 0.2

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.START_EPOCH = 0
_C.EPOCHS = 2
_C.WARMUP_EPOCHS = 5
_C.WEIGHT_DECAY = 0.05
_C.BASE_LR = 3e-5
_C.WARMUP_LR = 5e-7
_C.MIN_LR = 5e-6
# Clip gradient norm
_C.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be 
_C.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be 
_C.USE_CHECKPOINT = False

# LR scheduler

_C.SCHEDULER_NAME = 'plateau'
_C.FACTOR = 0.1
_C.PATIENCE = 5
# Epoch interval to decay LR, used in StepLRScheduler
_C.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.DECAY_RATE = 0.1

# Optimizer
_C.OPTIM_NAME = 'adamw'
# Optimizer Epsilon
_C.EPS = 1e-8
# Optimizer Betas
_C.BETAS = (0.9, 0.999)
# SGD momentum
_C.MOMENTUM = 0.9

_C.AMP_ENABLE = False 
_C.OUTPUT = 'output'
_C.SEED = 42
_C.EVAL_MODE = False
_C.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_C.OUTPUT_DIM = None 

def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config