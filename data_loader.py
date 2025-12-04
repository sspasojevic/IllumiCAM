"""
Data loading utilities for illuminant estimation.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import (
    DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, MEAN, STD
)


def get_transforms():
    """
    Get data transforms for training and validation/test.
    
    Returns:
        train_transform: Transform for training data
        val_test_transform: Transform for validation/test data
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
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        label_names: List of class names
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
    
    Args:
        train_dataset: Training dataset (if None, will load from disk)
        val_dataset: Validation dataset (if None, will load from disk)
        test_dataset: Test dataset (if None, will load from disk)
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    if train_dataset is None or val_dataset is None or test_dataset is None:
        train_dataset, val_dataset, test_dataset, _ = get_datasets()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

