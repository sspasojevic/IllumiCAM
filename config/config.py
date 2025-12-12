"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Centralized Configuration Module

This module contains all configuration constants, paths, and model registries
for the illuminant estimation project. It serves as the single source of truth
for device selection, file paths, training hyperparameters, and model definitions.

Exports:
    - DEVICE: Auto-detected compute device (CUDA, MPS, or CPU)
    - All paths: Data, models, output directories
    - Training config: Hyperparameters, batch sizes, learning rates
    - Model registry: MODELS, MODEL_PATHS, TARGET_LAYERS
"""
import os
import torch

# ============ PROJECT & DEVICE ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ============ PATHS ============
# Data
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
LSMI_TEST_PACKAGE = os.path.join(DATA_ROOT, "LSMI_Test_Package")
LSMI_IMAGES_DIR = os.path.join(LSMI_TEST_PACKAGE, "images")
LSMI_MASKS_DIR = os.path.join(LSMI_TEST_PACKAGE, "masks")
LSMI_DATASET_ROOT = os.path.join(DATA_ROOT, "LSMI")

# Models
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_PATHS = {
    'standard': os.path.join(SAVED_MODELS_DIR, 'best_illuminant_cnn_val_8084.pth'),
    'confidence': os.path.join(SAVED_MODELS_DIR, 'best_illuminant_cnn_confidence.pth'),
    'paper': os.path.join(SAVED_MODELS_DIR, 'best_paper_model.pth'),
    'illumicam3': os.path.join(SAVED_MODELS_DIR, 'best_illumicam3.pth')
}

# Output
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
CLUSTER_CENTERS_PATH = os.path.join(PROJECT_ROOT, "cluster_centers.npy")

# Info
INFO_DIR = os.path.join(DATA_ROOT, "info", "Info")
NIKON_CCM_MAT = os.path.join(INFO_DIR, "reference_wps_ccms_nikond810.mat")

# ============ TRAINING CONFIG ============
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_CLASSES = 5
DROPOUT_RATE = 0.25

SCHEDULER_FACTOR = 0.2
SCHEDULER_PATIENCE = 4
SCHEDULER_MIN_LR = 5e-12

PAPER_BATCH_SIZE = 100
PAPER_MOMENTUM = 0.9
PAPER_WEIGHT_DECAY = 0.0005
PAPER_LEARNING_RATE = 0.001

# ============ DATA CONFIG ============
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 8

# ============ MODEL REGISTRY ============
from src.models import IlluminantCNN, ConfidenceWeightedCNN, ColorConstancyCNN, IllumiCam3

MODELS = {
    'standard': IlluminantCNN,
    'confidence': ConfidenceWeightedCNN,
    'paper': lambda: ColorConstancyCNN(K=NUM_CLASSES, pretrained=False),
    'illumicam3': IllumiCam3
}

TARGET_LAYERS = {
    'standard': lambda model: model.conv5,
    'confidence': lambda model: model.conv5,
    'illumicam3': lambda model: model.conv5,
    'paper': lambda model: model.features[10]
}

