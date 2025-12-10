"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

LSMI Mask Evaluation Script

Evaluates illuminant estimation models against ground truth masks from the LSMI
test dataset. Generates CAM heatmaps for each model and computes spatial metrics
(IOU, DICE, MAE) by comparing predicted attention maps with ground truth illuminant
masks. Supports single model evaluation or comprehensive matrix evaluation across
all model-CAM combinations.

Uses:
    - config.config for paths and device
    - src.utils for model loading, CAM generation, and metrics
    - pytorch_grad_cam for Class Activation Mapping
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config import DEVICE, LSMI_IMAGES_DIR, LSMI_MASKS_DIR, MODEL_PATHS
from src.utils import (
    load_model, create_cam, ModelWrapper, calculate_metrics,
    process_raw_image, load_mask, CLUSTER_NAMES
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_CHOICES = list(MODEL_PATHS.keys())
CAM_CHOICES = ['gradcam', 'gradcam++', 'scorecam']


def evaluate_single_model(model_name, cam_method, gt_threshold=0.5, pred_threshold=0.5):
    """
    Evaluate a single model-CAM combination against ground truth masks.
    
    Args:
        model_name: Model type ('standard', 'confidence', 'paper', 'illumicam3')
        cam_method: CAM algorithm ('gradcam', 'gradcam++', 'scorecam')
        gt_threshold: Binarization threshold for ground truth masks
        pred_threshold: Binarization threshold for predicted masks
        
    Returns:
        Dictionary with metrics per cluster and overall averages, or None if error
    """
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None
    
    try:
        cam = create_cam(model, model_name, cam_method)
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
    if not os.path.exists(LSMI_IMAGES_DIR):
        print(f"Error: Images directory not found: {LSMI_IMAGES_DIR}")
        return None
    
    test_scenes = [f.replace(".nef", "") for f in os.listdir(LSMI_IMAGES_DIR) if f.endswith(".nef")]
    print(f"Evaluating on {len(test_scenes)} scenes...")
    
    for scene_id in tqdm(test_scenes, desc=f"{model_name}-{cam_method}"):
        try:
            # Load Image
            img_path = os.path.join(LSMI_IMAGES_DIR, f"{scene_id}.nef")
            
            # Load Raw for Model Input (Linear RGB)
            img_raw = process_raw_image(img_path, srgb=False)
            
            # Load sRGB for Visualization/Mask Reference
            img_rgb = process_raw_image(img_path, srgb=True)
            
            # Prepare for Model (Use Raw)
            img_pil = Image.fromarray(img_raw)
            input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            
            # Load GT Mask
            mask_path = os.path.join(LSMI_MASKS_DIR, f"{scene_id}_mask.npy")
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
                metrics = calculate_metrics(gt_channel, pred_mask, gt_threshold=gt_threshold, pred_threshold=pred_threshold)
                
                scene_metrics[f'{cluster_name}_IOU'] = metrics['iou']
                scene_metrics[f'{cluster_name}_DICE'] = metrics['dice']
                scene_metrics[f'{cluster_name}_MAE'] = metrics['mae']
                
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
    
    for model_name in MODEL_CHOICES:
        print(f"\nEvaluating Model: {model_name}")
        
        for cam_method in CAM_CHOICES:
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
    parser.add_argument('--model', type=str, default='standard', 
                       choices=MODEL_CHOICES,
                       help='Model to evaluate')
    parser.add_argument('--cam', type=str, default='gradcam++',
                       choices=CAM_CHOICES,
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

