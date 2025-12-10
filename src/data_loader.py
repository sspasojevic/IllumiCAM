"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Data Loading Module

Provides data loading utilities for illuminant estimation training and evaluation.
Creates PyTorch DataLoaders with appropriate transforms for train/val/test splits.
Handles image preprocessing (resize, normalization) using ImageNet statistics.

Uses:
    - config.config for dataset paths and hyperparameters
    - torchvision for data transforms and ImageFolder datasets
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config import DATASET_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, MEAN, STD, DEVICE


def get_transforms():
    """
    Get data transforms for training and validation/test splits.
    
    Returns:
        Tuple of (train_transform, val_test_transform)
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
    Load train, validation, and test datasets from ImageFolder structure.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_names)
    """
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "train"), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "val"), 
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "test"), 
        transform=val_test_transform
    )
    
    label_names = train_dataset.classes
    
    return train_dataset, val_dataset, test_dataset, label_names


def get_dataloaders(train_dataset=None, val_dataset=None, test_dataset=None, 
                    batch_size=None, num_workers=None):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset (creates new if None)
        val_dataset: Validation dataset (creates new if None)
        test_dataset: Test dataset (creates new if None)
        batch_size: Batch size (uses config default if None)
        num_workers: Number of workers (uses config default if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_names)
    """
    if train_dataset is None or val_dataset is None or test_dataset is None:
        train_dataset, val_dataset, test_dataset, _ = get_datasets()
    
    # Use defaults from config if not provided
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    # MPS optimization: pin_memory is often slow/buggy on Mac
    # Also, reduce num_workers if on MPS to avoid overhead
    is_mps = DEVICE.type == "mps"
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
