"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Unified Training Script for all model types.
Supports: standard, confidence, paper, illumicam3
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import configuration
from config.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR, VISUALIZATIONS_DIR, NUM_CLASSES, BATCH_SIZE,
    MODELS, MODEL_PATHS, SAVED_MODELS_DIR, PAPER_BATCH_SIZE, PAPER_MOMENTUM,
    PAPER_WEIGHT_DECAY, PAPER_LEARNING_RATE
)
from src.models.model import count_parameters
from src.data_loader import get_datasets, get_dataloaders


def train_one_epoch_standard(model, loader, criterion, optimizer, device):
    """
    Train for one epoch using standard cross-entropy loss.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Compute device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct/total})

    return running_loss / total, correct / total


def train_one_epoch_confidence(model, loader, criterion, optimizer, device):
    """
    Train confidence model for one epoch with entropy regularization.
    
    Args:
        model: Confidence-weighted model
        loader: Training data loader
        criterion: Classification loss function
        optimizer: Optimizer
        device: Compute device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, weights = model(images)
        
        # Classification loss
        cls_loss = criterion(outputs, labels)
        
        # Entropy regularization
        b, c, h, w = weights.shape
        entropy_loss = torch.sum(weights * torch.log(weights + 1e-8))
        lambda_entropy = 0.01
        
        loss = cls_loss + lambda_entropy * entropy_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct/total})

    return running_loss / total, correct / total


def validate_standard(model, loader, criterion, device):
    """
    Validate model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: Compute device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def validate_confidence(model, loader, criterion, device):
    """
    Validate confidence model for one epoch.
    
    Args:
        model: Confidence-weighted model
        loader: Validation data loader
        criterion: Loss function
        device: Compute device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def plot_training_curves(history, save_path, model_type):
    """
    Plot and save training/validation loss and accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Path to save the plot
        model_type: Model type name for title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')
    parser.add_argument('--model-type', type=str, 
                       choices=['standard', 'confidence', 'paper', 'illumicam3'],
                       required=True,
                       help='Model type to train')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config/default)')
    
    args = parser.parse_args()
    model_type = args.model_type
    
    print(f"Using device: {DEVICE}")
    print(f"\nTraining {model_type.upper()} model")
    
    # Determine batch size
    if args.batch_size:
        batch_size = args.batch_size
    elif model_type == 'paper':
        batch_size = PAPER_BATCH_SIZE
    else:
        batch_size = BATCH_SIZE
    
    print(f"Batch size: {batch_size}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, label_names = get_datasets()
    print(f"Classes: {label_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size
    )
    
    # Initialize model
    print(f"\nInitializing {model_type} model...")
    model_class = MODELS[model_type]
    model = model_class().to(DEVICE)
    print(model)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup loss, optimizer, scheduler
    if model_type == 'paper':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=PAPER_LEARNING_RATE,
            momentum=PAPER_MOMENTUM,
            weight_decay=PAPER_WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
        )
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
        )
    
    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    model_path = MODEL_PATHS[model_type]
    
    # Create saved_models directory if it doesn't exist
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        if model_type == 'confidence':
            t_loss, t_acc = train_one_epoch_confidence(
                model, train_loader, criterion, optimizer, DEVICE
            )
        else:
            t_loss, t_acc = train_one_epoch_standard(
                model, train_loader, criterion, optimizer, DEVICE
            )
        
        # Validate
        if model_type == 'confidence':
            v_loss, v_acc = validate_confidence(
                model, val_loader, criterion, DEVICE
            )
        else:
            v_loss, v_acc = validate_standard(
                model, val_loader, criterion, DEVICE
            )
        
        scheduler.step(v_loss)
        
        # Record history
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")
        
        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (val_acc: {v_acc:.4f})")
    
    # Plot training curves
    curves_path = os.path.join(VISUALIZATIONS_DIR, f"training_curves_{model_type}.png")
    plot_training_curves(history, curves_path, model_type)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
