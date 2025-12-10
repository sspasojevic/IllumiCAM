"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Core Utilities Module

This module provides essential utility functions for model loading, CAM generation,
evaluation metrics, and RAW image processing. Eliminates code duplication across
multiple scripts by centralizing commonly used operations.

Functions:
    - load_model(): Load any model with weights
    - create_cam(): Create CAM instance for any model
    - calculate_metrics(): Compute IOU, DICE, MAE metrics
    - process_raw_image(): Process NEF RAW files
    - load_mask(): Load ground truth masks

Classes:
    - ModelWrapper: Compatibility wrapper for pytorch_grad_cam
"""
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import rawpy
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from config.config import DEVICE, MODELS, MODEL_PATHS, TARGET_LAYERS, NUM_CLASSES

def load_model(model_name, weights_path=None, eval_mode=True, device=None):
    """
    Load a pre-trained model with weights.
    
    Args:
        model_name: Model type ('standard', 'confidence', 'paper', 'illumicam3')
        weights_path: Path to weights file (uses default from MODEL_PATHS if None)
        eval_mode: Set model to evaluation mode if True
        device: Target device (uses DEVICE from config if None)
    
    Returns:
        Loaded PyTorch model on specified device
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if device is None:
        device = DEVICE
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = MODELS[model_name]
    model = model_class() if model_name == 'paper' else model_class(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    if weights_path is None:
        weights_path = MODEL_PATHS.get(model_name)
    
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded: {weights_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if eval_mode:
        model.eval()
    
    return model

# ============ CAM UTILITIES ============

class ModelWrapper(nn.Module):
    """
    Wrapper to make models compatible with pytorch_grad_cam library.
    
    Handles different model output formats (some return tuples, others return logits directly).
    
    Args:
        model: The PyTorch model to wrap
        model_name: Model type identifier
    """
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name
    
    def forward(self, x):
        """Forward pass that extracts logits from model output."""
        if self.model_name in ['confidence', 'ConfidenceWeightedCNN']:
            logits, _ = self.model(x)
            return logits
        return self.model(x)

def create_cam(model, model_name, cam_method='gradcam'):
    """
    Create a Class Activation Map (CAM) instance for a model.
    
    Args:
        model: PyTorch model
        model_name: Model type ('standard', 'confidence', 'paper', 'illumicam3')
        cam_method: CAM algorithm ('gradcam', 'gradcam++', 'scorecam')
    
    Returns:
        CAM instance from pytorch_grad_cam library
        
    Raises:
        ValueError: If cam_method is not recognized
    """
    wrapper = ModelWrapper(model, model_name)
    target_layer = TARGET_LAYERS[model_name](model)
    
    cam_classes = {
        'gradcam': GradCAM,
        'GradCAM': GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'GradCAMPlusPlus': GradCAMPlusPlus,
        'scorecam': ScoreCAM,
        'ScoreCAM': ScoreCAM
    }
    
    if cam_method not in cam_classes:
        raise ValueError(f"Unknown CAM: {cam_method}")
    
    return cam_classes[cam_method](model=wrapper, target_layers=[target_layer])

# ============ METRICS ============

def calculate_iou(gt_mask, pred_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IOU) for binary masks.
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        threshold: Binarization threshold
        
    Returns:
        IOU score (0.0 to 1.0)
    """
    gt_bin = (gt_mask > threshold).astype(bool)
    pred_bin = (pred_mask > threshold).astype(bool)
    
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    
    return (intersection / union) if union > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_dice(gt_mask, pred_mask, threshold=0.5):
    """
    Calculate DICE coefficient for binary masks.
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        threshold: Binarization threshold
        
    Returns:
        DICE score (0.0 to 1.0)
    """
    gt_bin = (gt_mask > threshold).astype(bool)
    pred_bin = (pred_mask > threshold).astype(bool)
    
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    dice_denom = gt_bin.sum() + pred_bin.sum()
    
    return (2 * intersection / dice_denom) if dice_denom > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_mae(gt_mask, pred_mask):
    """
    Calculate Mean Absolute Error (MAE) between masks.
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        
    Returns:
        MAE score
    """
    return np.mean(np.abs(gt_mask - pred_mask))

def calculate_metrics(gt_mask, pred_mask, gt_threshold=0.5, pred_threshold=0.5):
    """
    Calculate all evaluation metrics (IOU, DICE, MAE) for mask comparison.
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        gt_threshold: Threshold for ground truth binarization
        pred_threshold: Threshold for prediction binarization
    
    Returns:
        Dictionary with keys 'iou', 'dice', 'mae'
    """
    return {
        'iou': calculate_iou(gt_mask, pred_mask, max(gt_threshold, pred_threshold)),
        'dice': calculate_dice(gt_mask, pred_mask, max(gt_threshold, pred_threshold)),
        'mae': calculate_mae(gt_mask, pred_mask)
    }

# ============ LSMI UTILITIES ============

CLUSTER_NAMES = ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']

def get_cluster_names():
    """
    Get illuminant cluster names.
    
    Returns:
        List of cluster names: ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
    """
    return CLUSTER_NAMES

def process_raw_image(raw_path, srgb=False):
    """
    Process RAW image (NEF) and convert to RGB.
    
    Args:
        raw_path: Path to .nef file
        srgb: If True, returns sRGB (gamma corrected, white balanced).
              If False, returns linear RGB (raw sensor data).
                     
    Returns:
        RGB image array (uint8 if srgb=True, uint16 if srgb=False)
    """
    with rawpy.imread(raw_path) as raw:
        if srgb:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=False)
        else:
            rgb = raw.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], 
                                 no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
    return rgb

def load_mask(mask_path, target_shape=None):
    """
    Load ground truth mask and optionally resize to target shape.
    
    Args:
        mask_path: Path to .npy mask file
        target_shape: (Height, Width) tuple for resizing (optional)
        
    Returns:
        Mask array of shape (H, W, 5) with channels for each illuminant cluster
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
        
    mask = np.load(mask_path)
    
    if target_shape is not None:
        h, w = target_shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
    return mask
