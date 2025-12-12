"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Model Evaluation Script

Evaluates trained illuminant estimation models on the test dataset.
Computes classification metrics (accuracy, precision, recall, F1) and
generates confusion matrices for all supported model architectures.

Supports: standard, confidence, paper, illumicam3

Usage:
python evaluate_models.py --model-type model_name --batch-size batch_size

Example:
python evaluate_models.py --model-type standard
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config import DEVICE, VISUALIZATIONS_DIR, MODEL_PATHS
from src.utils import load_model
from src.data_loader import get_datasets, get_dataloaders


def evaluate_test_set(model_type='standard', batch_size=256):
    """
    Evaluate model on test dataset and compute metrics.
    
    Args:
        model_type: Type of model ('standard', 'confidence', 'paper', 'illumicam3')
        batch_size: Batch size for evaluation
    
    Returns:
        all_preds: All predictions
        all_labels: All true labels
        label_names: Class names
        test_acc: Test accuracy
    """
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Get model path for the specified model type
    model_path = MODEL_PATHS.get(model_type)
    if model_path is None:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(MODELS.keys())}")
    
    # Load datasets
    train_dataset, val_dataset, test_dataset, label_names = get_datasets()
    
    # Create test loader
    _, _, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=batch_size
    )
    
    # Load model
    print(f"Loading {model_type} model from {model_path}...")
    model = load_model(model_type, weights_path=model_path, eval_mode=True, device=DEVICE)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Handle models that return tuples (confidence model)
            if model_type == 'confidence':
                outputs, _ = model(images)
            else:
                outputs = model(images)
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_type.upper()}")
    plt.tight_layout()
    
    # Save in visualizations/evaluate folder
    evaluate_dir = os.path.join(VISUALIZATIONS_DIR, "evaluate")
    os.makedirs(evaluate_dir, exist_ok=True)
    save_path = os.path.join(evaluate_dir, f"confusion_matrix_{model_type}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix to {save_path}")
    
    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    return all_preds, all_labels, label_names, test_acc


def main():
    """
    Main evaluation function.
    """

    parser = argparse.ArgumentParser(description='Evaluate illuminant estimation model')
    parser.add_argument('--model-type', type=str, 
                       choices=['standard', 'confidence', 'paper', 'illumicam3'],
                       default='standard',
                       help='Model type to evaluate')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    evaluate_test_set(
        model_type=args.model_type,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

