import torch


# parameters config
seed = 12138
torch.manual_seed(seed)

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # paper used 64
WEIGHT_DECAY = 0     # paper used 5e-4
EPOCHS = 135
IOU_THRESHOLD = 0.5
PROB_THRESHOLD = 0.4
IMG_DIR = "VOCdevkit/img/"
TAR_DIR = "VOCdevkit/tar/"
LOAD_MODEL = False
LOAD_MODEL_FILE = "train0w"
S = 7   # paper used 7 so that an image is divided into a 7X7 grid
