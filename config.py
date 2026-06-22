import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
LR = 0.0002
Z_DIM = 100
EPOCHS = 30

IMAGE_SIZE = 28
CHANNELS = 1