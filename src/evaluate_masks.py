"""
LSMI Mask Evaluation Script

This script evaluates the IlluminantCNN model on the balanced LSMI test set.
It performs the following:
1. Loads the pre-trained model.
2. Runs inference on the test images.
3. Generates Grad-CAM heatmaps for each illuminant cluster.
4. Calculates metrics (IOU, DICE, P-MAE) against the Ground Truth masks.
5. Outputs results summary.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import Models
from src.models.model import IlluminantCNN
from src.models.model_confidence import ConfidenceWeightedCNN
from src.models.model_paper import ColorConstancyCNN
from src.models.model_illumicam3 import IllumiCam3

# Import CAM Methods
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Import Test Package Utils
from src.lsmi_utils import process_raw_image, load_mask, CLUSTER_NAMES

# Hardcoded configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Paths
TEST_PACKAGE_DIR = os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package")
IMAGES_DIR = os.path.join(TEST_PACKAGE_DIR, "images")
MASKS_DIR = os.path.join(TEST_PACKAGE_DIR, "masks")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Model registry
MODELS = {
    "IlluminantCNN": IlluminantCNN,
    "IllumiCam3": IllumiCam3,
    "ConfidenceWeightedCNN": ConfidenceWeightedCNN,
    "ColorConstancyCNN": ColorConstancyCNN
}

# Model paths
MODEL_PATHS = {
    "IlluminantCNN": os.path.join(SAVED_MODELS_DIR, "best_illuminant_cnn_val_8084.pth"),
    "IllumiCam3": os.path.join(SAVED_MODELS_DIR, "best_illumicam3.pth"),
    "ColorConstancyCNN": os.path.join(SAVED_MODELS_DIR, "best_paper_model.pth"),
    "ConfidenceWeightedCNN": os.path.join(SAVED_MODELS_DIR, "best_illuminant_cnn_confidence.pth")
}

CAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "ScoreCAM": ScoreCAM
}


class ModelWrapper(torch.nn.Module):
    """Wrapper to make models compatible with pytorch_grad_cam."""
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name
    
    def forward(self, x):
        if self.model_name == "ConfidenceWeightedCNN":
            logits, _ = self.model(x)
            return logits
        else:
            # Other models return logits directly
            return self.model(x)


def load_model(model_name, model_path):
    """Load a pre-trained model."""
    print(f"Loading model architecture: {model_name}")
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    
    if model_name == "ColorConstancyCNN":
        model = model_class(K=5, pretrained=False)
    else:
        model = model_class(num_classes=5)
        
    model = model.to(DEVICE)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random/initialized weights.")
    else:
        print(f"Model file not found at {model_path}. Using random/initialized weights.")
        
    model.eval()
    return model


def get_cam(model, model_name, method_name):
    """Get CAM instance for a model."""
    if method_name not in CAM_METHODS:
        raise ValueError(f"Unknown CAM method: {method_name}. Available: {list(CAM_METHODS.keys())}")
    
    # Wrap model for CAM compatibility
    model_wrapper = ModelWrapper(model, model_name)
    
    # Determine target layers based on model architecture
    if model_name == "IlluminantCNN":
        target_layers = [model.conv5]
    elif model_name == "IllumiCam3":
        target_layers = [model.conv5]
    elif model_name == "ConfidenceWeightedCNN":
        target_layers = [model.conv5]
    elif model_name == "ColorConstancyCNN":
        # Target the last convolutional layer of AlexNet features
        target_layers = [model.features[10]]
    else:
        target_layers = [list(model.modules())[-1]]  # Fallback

    cam_class = CAM_METHODS[method_name]
    return cam_class(model=model_wrapper, target_layers=target_layers)


def calculate_metrics(gt_mask, pred_mask, gt_threshold=0.5, pred_threshold=0.5):
    """
    Calculates IOU, DICE, and MAE.
    IOU and DICE are calculated on binary masks created by thresholding GT and Pred.
    MAE is calculated on continuous values.
    """
    # MAE (on continuous values)
    mae = np.mean(np.abs(gt_mask - pred_mask))
    
    # Threshold for IOU/DICE
    gt_bin = (gt_mask > gt_threshold).astype(bool)
    pred_bin = (pred_mask > pred_threshold).astype(bool)
    
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
        
    dice_denom = gt_bin.sum() + pred_bin.sum()
    if dice_denom == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = 2 * intersection / dice_denom
        
    return iou, dice, mae


def evaluate_single_model(model_name, cam_method, gt_threshold=0.5, pred_threshold=0.5):
    """Evaluate a single model-CAM combination."""
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None:
        print(f"Warning: No model path found for {model_name}")
        return None
    
    # Load model
    try:
        model = load_model(model_name, model_path)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None
    
    # Setup CAM
    try:
        cam = get_cam(model, model_name, cam_method)
    except Exception as e:
        print(f"Failed to init {cam_method}: {e}")
        return None
    
    print(f"Using CAM Method: {cam_method} on {model_name}")
    
    # Transforms for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    # Get list of test scenes
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found: {IMAGES_DIR}")
        return None
    
    test_scenes = [f.replace(".nef", "") for f in os.listdir(IMAGES_DIR) if f.endswith(".nef")]
    print(f"Evaluating on {len(test_scenes)} scenes...")
    
    for scene_id in tqdm(test_scenes, desc=f"{model_name}-{cam_method}"):
        try:
            # Load Image
            img_path = os.path.join(IMAGES_DIR, f"{scene_id}.nef")
            
            # Load Raw for Model Input (Linear RGB)
            img_raw = process_raw_image(img_path, srgb=False)
            
            # Load sRGB for Visualization/Mask Reference
            img_rgb = process_raw_image(img_path, srgb=True)
            
            # Prepare for Model (Use Raw)
            img_pil = Image.fromarray(img_raw)
            input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            
            # Load GT Mask
            mask_path = os.path.join(MASKS_DIR, f"{scene_id}_mask.npy")
            gt_mask = load_mask(mask_path, target_shape=img_rgb.shape)
            
            # Generate CAM for each cluster
            scene_metrics = {'scene': scene_id}
            
            for i, cluster_name in enumerate(CLUSTER_NAMES):
                # Get GT channel
                gt_channel = gt_mask[:, :, i]
                
                # Skip if GT is empty
                if gt_channel.max() == 0:
                    continue
                
                targets = [ClassifierOutputTarget(i)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
                # Resize CAM to original image size
                pred_mask = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
                
                # Normalize pred_mask to [0, 1] if needed
                if pred_mask.max() > 1.0:
                    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
                
                # Calculate Metrics
                iou, dice, mae = calculate_metrics(gt_channel, pred_mask, gt_threshold=gt_threshold, pred_threshold=pred_threshold)
                
                scene_metrics[f'{cluster_name}_IOU'] = iou
                scene_metrics[f'{cluster_name}_DICE'] = dice
                scene_metrics[f'{cluster_name}_MAE'] = mae
                
            results.append(scene_metrics)
            
        except Exception as e:
            print(f"Error processing {scene_id}: {e}")
    
    return pd.DataFrame(results)


def print_summary(df_results):
    """Print summary statistics."""
    if df_results is None or df_results.empty:
        print("No results to summarize.")
        return
    
    # Calculate averages
    summary = {}
    for metric in ['IOU', 'DICE', 'MAE']:
        for cluster in CLUSTER_NAMES:
            col = f'{cluster}_{metric}'
            if col in df_results.columns:
                summary[col] = df_results[col].mean()
    
    print("\nAverage Metrics:")
    for cluster in CLUSTER_NAMES:
        print(f"\nCluster: {cluster}")
        print(f"  IOU:  {summary.get(f'{cluster}_IOU', 0):.4f}")
        print(f"  DICE: {summary.get(f'{cluster}_DICE', 0):.4f}")
        print(f"  MAE:  {summary.get(f'{cluster}_MAE', 0):.4f}")
    
    # Overall Average
    print("\nOverall Averages:")
    ious = [summary.get(f'{c}_IOU', 0) for c in CLUSTER_NAMES]
    dices = [summary.get(f'{c}_DICE', 0) for c in CLUSTER_NAMES]
    maes = [summary.get(f'{c}_MAE', 0) for c in CLUSTER_NAMES]
    
    if ious:
        print(f"  mIOU: {np.mean(ious):.4f}")
    if dices:
        print(f"  mDICE: {np.mean(dices):.4f}")
    if maes:
        print(f"  mMAE: {np.mean(maes):.4f}")


def evaluate_matrix(gt_threshold=0.0, pred_threshold=0.1):
    """Run comprehensive evaluation matrix on all model-CAM combinations."""
    print("Starting Comprehensive Evaluation Matrix...")
    print(f"GT Threshold: {gt_threshold}, Pred Threshold: {pred_threshold}")
    
    matrix_results = []
    
    for model_name in MODEL_PATHS.keys():
        print(f"\nEvaluating Model: {model_name}")
        
        for cam_method in CAM_METHODS.keys():
            print(f"  Using CAM: {cam_method}")
            
            df_results = evaluate_single_model(model_name, cam_method, gt_threshold, pred_threshold)
            
            if df_results is None or df_results.empty:
                print(f"    -> No results for {model_name}-{cam_method}")
                continue
            
            # Calculate per-scene averages
            scene_ious = []
            scene_dices = []
            scene_maes = []
            
            for cluster in CLUSTER_NAMES:
                iou_col = f'{cluster}_IOU'
                dice_col = f'{cluster}_DICE'
                mae_col = f'{cluster}_MAE'
                
                if iou_col in df_results.columns:
                    scene_ious.extend(df_results[iou_col].dropna().tolist())
                if dice_col in df_results.columns:
                    scene_dices.extend(df_results[dice_col].dropna().tolist())
                if mae_col in df_results.columns:
                    scene_maes.extend(df_results[mae_col].dropna().tolist())
            
            res = {
                "Model": model_name,
                "CAM": cam_method,
                "mIOU": np.mean(scene_ious) if scene_ious else 0,
                "mDICE": np.mean(scene_dices) if scene_dices else 0,
                "mMAE": np.mean(scene_maes) if scene_maes else 0
            }
            matrix_results.append(res)
            print(f"    -> mIOU: {res['mIOU']:.4f}, mDICE: {res['mDICE']:.4f}, mMAE: {res['mMAE']:.4f}")
    
    # Display Results
    df_matrix = pd.DataFrame(matrix_results)
    print("\nEvaluation Matrix Results:")
    print(df_matrix.to_string(index=False))
    
    return df_matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate model CAM predictions against LSMI ground truth masks')
    parser.add_argument('--model', type=str, default='IlluminantCNN', 
                       choices=list(MODEL_PATHS.keys()),
                       help='Model to evaluate')
    parser.add_argument('--cam', type=str, default='GradCAMPlusPlus',
                       choices=list(CAM_METHODS.keys()),
                       help='CAM method to use')
    parser.add_argument('--gt_threshold', type=float, default=0.0,
                       help='Threshold for ground truth mask binarization')
    parser.add_argument('--pred_threshold', type=float, default=0.1,
                       help='Threshold for predicted mask binarization')
    parser.add_argument('--matrix', action='store_true',
                       help='Run comprehensive evaluation matrix on all model-CAM combinations')
    parser.add_argument('--matrix_gt_threshold', type=float, default=0.0,
                       help='GT threshold for matrix evaluation')
    parser.add_argument('--matrix_pred_threshold', type=float, default=0.1,
                       help='Pred threshold for matrix evaluation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path for results')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    if args.matrix:
        # Run comprehensive matrix
        df_matrix = evaluate_matrix(args.matrix_gt_threshold, args.matrix_pred_threshold)
        if args.output:
            df_matrix.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    else:
        # Run single model-CAM evaluation
        df_results = evaluate_single_model(args.model, args.cam, args.gt_threshold, args.pred_threshold)
        
        if df_results is not None and not df_results.empty:
            print_summary(df_results)
            
            if args.output:
                df_results.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

