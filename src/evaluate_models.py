"""
Evaluation script for illuminant estimation model.
Supports: standard, confidence, paper, illumicam3
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

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Hardcoded configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
NUM_CLASSES = 5

from src.models.model import IlluminantCNN
from src.models.model_confidence import ConfidenceWeightedCNN
from src.models.model_paper import ColorConstancyCNN
from src.models.model_illumicam3 import IllumiCam3
from src.data_loader import get_datasets, get_dataloaders

# Model registry
MODELS = {
    'standard': IlluminantCNN,
    'confidence': ConfidenceWeightedCNN,
    'paper': lambda: ColorConstancyCNN(K=NUM_CLASSES, pretrained=False),
    'illumicam3': IllumiCam3
}

# Model save paths
MODEL_PATHS = {
    'standard': os.path.join(SAVED_MODELS_DIR, 'best_illuminant_cnn_val_8084.pth'),
    'confidence': os.path.join(SAVED_MODELS_DIR, 'best_illuminant_cnn_confidence.pth'),
    'paper': os.path.join(SAVED_MODELS_DIR, 'best_paper_model.pth'),
    'illumicam3': os.path.join(SAVED_MODELS_DIR, 'best_illumicam3.pth')
}


def evaluate_test_set(model_type='standard', batch_size=256):
    """
    Evaluate the model on the test set.
    
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
    model_class = MODELS[model_type]
    model = model_class().to(DEVICE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
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
    """Main evaluation function."""
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

