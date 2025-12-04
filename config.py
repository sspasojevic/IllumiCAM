"""
Configuration file for illuminant estimation training and evaluation.
"""

import torch

# Data paths
DATA_ROOT = "dataset"
CLUSTER_CENTERS_PATH = "cluster_centers.npy"

# Training hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 8
IMG_SIZE = 224
NUM_CLASSES = 5

# Model paths
BEST_MODEL_PATH = "best_illuminant_cnn.pth"
BEST_MODEL_PATH_VAL = "best_illuminant_cnn_val_8084.pth"

# Image normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Output directories
VISUALIZATIONS_DIR = "visualizations"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scheduler settings
SCHEDULER_FACTOR = 0.2
SCHEDULER_PATIENCE = 4
SCHEDULER_MIN_LR = 5e-12

# Dropout
DROPOUT_RATE = 0.25

