"""
Data loading utilities for illuminant estimation.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# Hardcoded configuration
# Get project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
BATCH_SIZE = 64
NUM_WORKERS = 8
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms():
    """
    Get data transforms for training and validation/test.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    return train_transform, val_test_transform


def get_datasets():
    """
    Load train, validation, and test datasets.
    """
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, "train"), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, "val"), 
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, "test"), 
        transform=val_test_transform
    )
    
    label_names = train_dataset.classes
    
    return train_dataset, val_dataset, test_dataset, label_names


def get_dataloaders(train_dataset=None, val_dataset=None, test_dataset=None, 
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create DataLoaders for train, validation, and test sets.
    """
    if train_dataset is None or val_dataset is None or test_dataset is None:
        train_dataset, val_dataset, test_dataset, _ = get_datasets()
    
    # MPS optimization: pin_memory is often slow/buggy on Mac
    # Also, reduce num_workers if on MPS to avoid overhead
    is_mps = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    pin_memory = False if is_mps else True
    
    # On Mac, high worker count often degrades performance. Cap it.
    if is_mps:
        num_workers = 4  # Try 4 workers to hide disk latency

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
