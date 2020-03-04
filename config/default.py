import os 
from yacs.config import CfgNode as CN 

# -------------------------
# Config Definition
# -------------------------
_C = CN()
_C.DEVICE = "cuda:0"

# -------------------------
# Input Configuration
# -------------------------
_C.INPUT = CN()
# min size of sorter side of the image
_C.INPUT.MIN_SIZE = 256  # only valid for photo-realistic transfer (Fast Photo Style)
# Maximum size of longer side of the image
_C.INPUT.MAX_SIZE = 960  # only valid for photo-realistic transfer  (Fast Photo Style)
# Resize image so that its shorter side = FINE_SIZE
_C.INPUT.FINE_SIZE = 256

# -------------------------
# Dataset 
# -------------------------
_C.DATASET = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASET.TRAIN = ''
# List of the dataset names for validation, as present in paths_catalog.py
_C.DATASET.VAL = ''
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASET.TEST = ''

# -------------------------
# DataLoader
# -------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Batch Size
_C.DATALOADER.BATCH_SIZE = 8

# -------------------------
# Optimizer Configuration
# -------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'adam'

_C.OPTIMIZER.MAX_ITER = 40000

_C.OPTIMIZER.BASE_LR = 0.001

_C.OPTIMIZER.SGD_MOMENTUM = 0.9

_C.OPTIMIZER.ADADELTA_RHO = 0.95
_C.OPTIMIZER.ADADELTA_EPS = 1e-6

_C.OPTIMIZER.ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0.1

_C.OPTIMIZER.ADAM_BETAS = (0.9, 0.999)
_C.OPTIMIZER.ADAM_EPS = 1e-8

_C.OPTIMIZER.RMSPROP_DECAY = 0.9
_C.OPTIMIZER.RMSPROP_MOMENTUM = 0.9
_C.OPTIMIZER.RMSPROP_EPS = 1e-8

_C.OPTIMIZER.LR_SCHEDULER = CN()
_C.OPTIMIZER.LR_SCHEDULER.NAME = 'lambda_lr'

_C.OPTIMIZER.LR_SCHEDULER.LR_DECAY = 5e-5
_C.OPTIMIZER.LR_SCHEDULER.T_MULT = 1

# -------------------------
# Model Configuration
# -------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'AdaIN'
_C.MODEL.ENCODER_NAME = ('vgg19-pytorch', 'vgg_normalised_conv4_1')
_C.MODEL.DECODER_NAME = ('vgg19-pytorch', )
_C.MODEL.PARTS = ()
_C.MODEL.ALPHA = 1.0

# ------------------------
# AdaIN
# ------------------------
_C.MODEL.ADAIN = CN()
_C.MODEL.ADAIN.CONTENT_WEIGHT = 1.0
_C.MODEL.ADAIN.STYLE_WEIGHT = 10.0

# ------------------------
# Linear Style Transfer
# ------------------------
_C.MODEL.LST = CN()
_C.MODEL.LST.LAYER = 'r31'
_C.MODEL.LST.MATRIX_SIZE = 32
_C.MODEL.LST.STYLE_LAYERS = ("r11","r21","r31","r41")
_C.MODEL.LST.CONTENT_LAYERS = 'r41'
_C.MODEL.LST.STYLE_WEIGHT = 0.02
_C.MODEL.LST.CONTENT_WEIGHT = 1.
_C.MODEL.LST.SPN_NUM = 1
_C.MODEL.LST.DISABLE_TRANSFER = False
_C.MODEL.LST.LOAD_LOSS_NET = True
_C.MODEL.LST.DISABLE_SPN = False

# ------------------------
# Fast Photo Style
# ------------------------
_C.MODEL.FPS = CN()
_C.MODEL.FPS.R = 35
_C.MODEL.FPS.EPS = 0.001
_C.MODEL.FPS.BETA = 0.9999
_C.MODEL.FPS.FAST = True

# ------------------------
# Misc
# ------------------------
_C.INPUT_DIR = '/home/alan/Downloads/style_transfer'
_C.MODEL_DIR = '/home/alan/Downloads/style_transfer'
_C.OUTPUT_DIR = '/home/alan/Downloads/style_transfer'

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

