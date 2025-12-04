"""
Evaluation script for illuminant estimation model.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, BEST_MODEL_PATH_VAL, VISUALIZATIONS_DIR
from model import IlluminantCNN
from data_loader import get_datasets, get_dataloaders


def evaluate_test_set(model_path=BEST_MODEL_PATH_VAL, batch_size=256):
    """
    Evaluate the model on the test set.
    
    Args:
        model_path: Path to the saved model
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
    
    # Load datasets
    train_dataset, val_dataset, test_dataset, label_names = get_datasets()
    
    # Create test loader
    _, _, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=batch_size
    )
    
    # Load model
    model = IlluminantCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    save_path = os.path.join(VISUALIZATIONS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix to {save_path}")
    
    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    return all_preds, all_labels, label_names, test_acc


def main():
    """Main evaluation function."""
    evaluate_test_set()


if __name__ == "__main__":
    main()

