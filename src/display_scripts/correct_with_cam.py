#!/usr/bin/env python3
"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

CAM-Guided Spatial Color Correction Tool

Performs spatially-aware white balance correction guided by Class Activation Maps.
Processes RAW NEF images using model predictions to identify illuminant regions,
then applies appropriate white balance to each region. Uses softmax-weighted blending 
when multiple illuminants are detected.

Outputs visualization showing original, corrected, single illuminant assumption, 
ground truth mask cluster corrected, and ground truth mask true illuminant corrected.

Usage:
python correct_with_cam.py --image path/to/image.nef --model model_name --cam cam_method --output path/to/output_directory

Example:
python correct_with_cam.py --image Data/LSMI_Test_Package/images/Place101.nef --model standard --cam gradcam
"""

# Imports
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import textwrap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt display errors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.config import MEAN, STD, DEVICE, IMG_SIZE, VISUALIZATIONS_DIR, MODEL_PATHS, LSMI_MASKS_DIR, LSMI_TEST_PACKAGE, LSMI_IMAGES_DIR
from src.utils import load_model, create_cam, process_raw_image, load_mask, CLUSTER_NAMES
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_CHOICES = list(MODEL_PATHS.keys())
CAM_CHOICES = ['gradcam', 'gradcam++', 'scorecam']

TRAIN_LABELS = CLUSTER_NAMES


def angular_error(x, y):
    """
    Calculate angular error between two RGB vectors.
    
    Args:
        x: First RGB vector
        y: Second RGB vector
    
    Returns:
        Angular error in degrees
    """

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm < 1e-8 or y_norm < 1e-8:
        return 180.0
    x = x / x_norm
    y = y / y_norm
    dot = np.clip(np.dot(x, y), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def map_illuminant_to_cluster(illuminant_rgb, cluster_centers_dict):
    """
    Map the GT illuminant RGB to the closest cluster using angular error.
    
    Args:
        illuminant_rgb: RGB illuminant value
        cluster_centers_dict: Dictionary of cluster centers
    
    Returns:
        Tuple of (best_cluster_name, min_error)
    """

    illuminant_rgb = np.array(illuminant_rgb, dtype=np.float32)
    # Normalize to sum=1 (chromaticity)
    if illuminant_rgb.sum() > 1e-8:
        illuminant_rgb = illuminant_rgb / illuminant_rgb.sum()
    
    min_error = float('inf')
    best_cluster = None
    
    for cluster_name, cluster_center in cluster_centers_dict.items():
        cluster_center = np.array(cluster_center, dtype=np.float32)
        if cluster_center.sum() > 1e-8:
            cluster_center = cluster_center / cluster_center.sum()
        error = angular_error(illuminant_rgb, cluster_center)
        if error < min_error:
            min_error = error
            best_cluster = cluster_name
    
    return best_cluster, min_error

def load_cluster_centers():
    """
    Load cluster centers dict from npy file.
    
    Returns:
        Dictionary mapping cluster names to center values
    """

    path = os.path.join(PROJECT_ROOT, "cluster_centers.npy")
    centers = np.load(path, allow_pickle=True).item()
    return centers

def get_wps_for_clusters():
    """
    Get white points for each cluster from cluster_centers.npy.
    
    Returns:
        Dictionary mapping cluster names to white point dictionaries
    """

    centers = load_cluster_centers()
    cluster_wps = {}

    for name in TRAIN_LABELS:
        wp = centers[name]
        wp_norm = wp / (wp.sum() + 1e-12)
        cluster_wps[name] = {
            'wp': wp_norm
        }
    return cluster_wps

# ---------- Raw correction ----------
def apply_correction(raw_img, wp):
    """
    Apply white balance to linear raw image.
    
    Args:
        raw_img: Linear raw image
        wp: White point (3-vector)
    
    Returns:
        Corrected image array
    """

    raw = raw_img.astype(np.float32)

    if wp[1] == 0:
        wp_norm = wp / (wp.sum() + 1e-12)
    else:
        wp_norm = wp / (wp[1] + 1e-12)

    # Avoid divide-by-zero in wb
    wb = raw / (wp_norm[None, None, :] + 1e-12)
    wb = np.clip(wb, 0.0, None)

    # Scale to [0,1] using percentile
    pmax = np.percentile(wb, 99.5)
    if pmax > 0:
        rendered = wb / (pmax + 1e-12)
        rendered = np.clip(rendered, 0.0, 1.0)
    else:
        rendered = wb

    return rendered

def normalize_cam_to_0_1(cam):
    """
    Scale CAM to [0,1] by min/max.
    
    Args:
        cam: CAM array
    
    Returns:
        Normalized CAM array in range [0, 1]
    """

    mn, mx = float(cam.min()), float(cam.max())
    if mx - mn < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - mn) / (mx - mn)).astype(np.float32)

def smooth_mask(mask, ksize=41, sigma=0):
    """
    Smooth mask with Gaussian blur.
    
    Args:
        mask: Input mask array
        ksize: Kernel size for Gaussian blur
        sigma: Standard deviation for Gaussian blur
    
    Returns:
        Smoothed mask array
    """

    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma)

def cams_to_softmax_weights(cam_dict, eps=1e-8, temp=1.0):
    """
    Convert CAM dict to per-pixel softmax weights.
    
    Args:
        cam_dict: Dictionary mapping cluster names to CAM arrays
        eps: Small epsilon for numerical stability
        temp: Temperature parameter for softmax
    
    Returns:
        Tuple of (weights_dict, weights_stack) where weights_dict maps cluster names
        to weight arrays and weights_stack is a stacked array
    """

    names = list(cam_dict.keys())
    cams = np.stack([cam_dict[n] for n in names], axis=-1)  # H x W x C
    # optional temperature and exponentiation (keep values non-negative)
    cams_scaled = cams / (temp + 1e-12)
    exps = np.exp(cams_scaled)  # large cams become dominant
    denom = np.sum(exps, axis=-1, keepdims=True) + eps
    soft = exps / denom
    weights = {names[i]: soft[..., i].astype(np.float32) for i in range(len(names))}
    return weights, soft  # soft is HxWxC

def lin_to_srgb(linear_rgb):
    """
    Convert linear RGB to sRGB using standard gamma correction.
    
    Args:
        linear_rgb: Array in range [0, 1]
    
    Returns:
        sRGB array in range [0, 1]
    """

    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
    mask = linear_rgb <= 0.0031308
    srgb = np.where(
        mask,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0)

def view_as_linear(x):
    """
    Pass-through for linear RGB (just clip).
    
    Preserves the 'dark and green' raw appearance by avoiding gamma correction.
    
    Args:
        x: Linear RGB array
    
    Returns:
        Clipped array in range [0, 1]
    """

    return np.clip(x, 0.0, 1.0)

# ---------- Processing Functions ----------
def process_single_image(image_path, model, cluster_mapping, args):
    """
    Process a single image and return visualization data.
    
    Args:
        image_path: Path to input image
        model: PyTorch model
        cluster_mapping: Dictionary mapping clusters to white points
        args: Command line arguments
    
    Returns:
        Dictionary containing processed image data
    """

    # Load raw linear image for correction
    raw_linear = process_raw_image(image_path, srgb=False)
    
    # Normalize to [0, 1] range for apply_correction
    if raw_linear.dtype == np.uint8:
        raw_linear = raw_linear.astype(np.float32) / 255.0
    elif raw_linear.dtype == np.uint16:
        raw_linear = raw_linear.astype(np.float32) / 65535.0
    else:
        raw_linear = raw_linear.astype(np.float32)
        if raw_linear.max() > 1.0:
            raw_linear = raw_linear / raw_linear.max()
    
    raw_linear = np.clip(raw_linear, 0.0, 1.0)

    # Prepare model input
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Load image for model input
    if image_path.lower().endswith('.nef'):
        rgb_array = process_raw_image(image_path, srgb=False)
        pil_img = Image.fromarray(rgb_array)
    else:
        pil_img = Image.open(image_path).convert('RGB')

    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Get CAMs
    cam = create_cam(model, args.model, args.cam)

    # Get logits/probs and predicted class
    with torch.no_grad():
        outputs = model(img_tensor) if args.model != 'confidence' else model(img_tensor)[0]
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = int(outputs.argmax(dim=1)[0].item())
        pred_class = TRAIN_LABELS[pred_idx]

    # Generate CAM for each cluster
    cams_raw = {}
    for i, class_name in enumerate(TRAIN_LABELS):
        targets = [ClassifierOutputTarget(i)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        grayscale_cam = normalize_cam_to_0_1(grayscale_cam)
        cam_resized = cv2.resize(grayscale_cam, (raw_linear.shape[1], raw_linear.shape[0]), interpolation=cv2.INTER_LINEAR)
        cams_raw[class_name] = cam_resized

    # Smooth cams
    cams_norm = {}
    for k, v in cams_raw.items():
        n = normalize_cam_to_0_1(v)
        s = smooth_mask(n, ksize=args.smooth_ksize)
        s = normalize_cam_to_0_1(s)
        cams_norm[k] = s

    # Convert cams -> per-pixel softmax weights
    weights_dict, weights_stack = cams_to_softmax_weights(cams_norm, temp=args.temp)

    # Precompute corrected images per cluster
    corrected_per_cluster = {}
    for name in TRAIN_LABELS:
        wp = cluster_mapping[name]['wp']
        corrected_per_cluster[name] = apply_correction(raw_linear, wp)

    # Compute base correction
    base_correction = corrected_per_cluster[pred_class]

    # Blend: weighted sum across clusters
    accumulated = np.zeros_like(raw_linear, dtype=np.float32)
    for i, name in enumerate(TRAIN_LABELS):
        w = weights_stack[..., i]
        w3 = w[..., None]
        accumulated += corrected_per_cluster[name] * w3

    weight_sum = weights_stack.sum(axis=-1)
    zero_mask = (weight_sum < 1e-6)[..., None]
    final_image = accumulated.copy()
    final_image[zero_mask.squeeze(-1)] = base_correction[zero_mask.squeeze(-1)]

    # Compute GT mask correction if available
    gt_corrected_srgb = None
    gt_clusters = None
    gt_oracle_srgb = None
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(LSMI_MASKS_DIR, f"{img_name}.npy")
    meta_path = os.path.join(LSMI_TEST_PACKAGE, "meta.json")
    
    if os.path.exists(mask_path):
        gt_mask = load_mask(mask_path, target_shape=raw_linear.shape[:2])
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            if img_name in meta_data:
                meta = meta_data[img_name]
                num_lights = meta.get("NumOfLights", 2)
                
                if gt_mask.shape[2] == num_lights:
                    cluster_centers = load_cluster_centers()
                    light_clusters = []
                    
                    for light_num in range(1, num_lights + 1):
                        light_key = f"Light{light_num}"
                        if light_key not in meta:
                            continue
                        light_rgb = np.array(meta[light_key], dtype=np.float32)
                        cluster_name, _ = map_illuminant_to_cluster(light_rgb, cluster_centers)
                        light_clusters.append(cluster_name)
                    
                    if len(light_clusters) == num_lights:
                        # GT cluster correction
                        gt_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
                        for i, cluster_name in enumerate(light_clusters):
                            weight = gt_mask[:, :, i]
                            weight_3d = weight[..., None]
                            gt_accumulated += corrected_per_cluster[cluster_name] * weight_3d
                        
                        gt_total_weight = gt_mask.sum(axis=-1, keepdims=True)
                        gt_mask_valid = gt_total_weight > 1e-6
                        gt_final = np.where(gt_mask_valid, gt_accumulated / (gt_total_weight + 1e-12), base_correction)
                        gt_corrected_srgb = lin_to_srgb(gt_final)
                        gt_clusters = light_clusters
                        
                        # GT oracle correction
                        gt_oracle_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
                        for i in range(num_lights):
                            light_key = f"Light{i+1}"
                            if light_key not in meta:
                                continue
                            light_rgb = np.array(meta[light_key], dtype=np.float32)
                            if light_rgb.sum() > 1e-8:
                                light_wp = light_rgb / light_rgb.sum()
                            else:
                                light_wp = light_rgb
                            
                            light_corrected = apply_correction(raw_linear, light_wp)
                            weight = gt_mask[:, :, i]
                            weight_3d = weight[..., None]
                            gt_oracle_accumulated += light_corrected * weight_3d
                        
                        gt_oracle_final = np.where(gt_mask_valid, gt_oracle_accumulated / (gt_total_weight + 1e-12), base_correction)
                        gt_oracle_srgb = lin_to_srgb(gt_oracle_final)
    
    # Convert to sRGB for visualization
    final_srgb = lin_to_srgb(final_image)
    base_srgb = lin_to_srgb(base_correction)
    # Load original image as sRGB using camera white balance
    original_srgb = process_raw_image(image_path, srgb=True).astype(np.float32) / 255.0
    
    return {
        'img_name': img_name,
        'original': original_srgb,
        'cam_corrected': final_srgb,
        'base': base_srgb,
        'gt_cluster': gt_corrected_srgb,
        'gt_oracle': gt_oracle_srgb,
        'gt_clusters': gt_clusters
    }

def wrap_title(title, max_length=20):
    """
    Wrap title to max 2 lines if it's too long.
    
    Args:
        title: Title string to wrap
        max_length: Maximum length per line
    
    Returns:
        Wrapped title string
    """

    if '\n' in title:
        lines = title.split('\n', 1)
        first_line = lines[0]
        if len(first_line) > max_length:
            words = first_line.split()
            if len(words) <= 1:
                first_line = f"{first_line[:max_length]}\n{first_line[max_length:]}"
            else:
                mid = len(words) // 2
                first_line = f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}"
        return f"{first_line}\n{lines[1]}" if len(lines) > 1 else first_line
    
    if len(title) <= max_length:
        return title

    words = title.split()
    if len(words) <= 1:
        
        return f"{title[:max_length]}\n{title[max_length:]}"

    # Find split point
    mid = len(words) // 2
    line1 = ' '.join(words[:mid])
    line2 = ' '.join(words[mid:])
    return f"{line1}\n{line2}"

def create_batch_visualization(all_results, args):
    """
    Create a grid visualization for multiple images.
    
    Args:
        all_results: List of result dictionaries from process_single_image
        args: Command line arguments
    """

    num_images = len(all_results)
    
    has_gt_cluster = any(r['gt_cluster'] is not None for r in all_results)
    has_gt_oracle = any(r['gt_oracle'] is not None for r in all_results)
    
    cols_per_image = 3  # original, cam corrected, base
    if has_gt_cluster:
        cols_per_image = 4
    if has_gt_oracle:
        cols_per_image = 5
    
    images_per_column = (num_images + 1) // 2  # Divide by 2, round up
    num_rows = images_per_column
    num_image_columns = 2
    
    cols = num_image_columns * cols_per_image
    rows = num_rows
    
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    
    for img_idx, result in enumerate(all_results):
        image_col = img_idx % num_image_columns
        row = img_idx // num_image_columns
        col_start = image_col * cols_per_image
        
        ax_first = plt.subplot(rows, cols, row * cols + col_start + 1)
        ax_first.imshow(result['original'])
        if row == 0:
            ax_first.set_title(wrap_title("Original"), fontsize=9)
        ax_first.axis('off')
        
        # CAM Corrected
        ax = plt.subplot(rows, cols, row * cols + col_start + 2)
        ax.imshow(result['cam_corrected'])
        if row == 0:
            ax.set_title(wrap_title("CAM Corrected"), fontsize=9)
        ax.axis('off')
        
        # Base
        ax = plt.subplot(rows, cols, row * cols + col_start + 3)
        ax.imshow(result['base'])
        if row == 0:
            ax.set_title(wrap_title("Single Illuminant Assumption"), fontsize=9)
        ax.axis('off')
        
        # GT Cluster
        if cols_per_image >= 4 and result['gt_cluster'] is not None:
            ax = plt.subplot(rows, cols, row * cols + col_start + 4)
            ax.imshow(result['gt_cluster'])
            if row == 0:
                title = "GT Mask Cluster Corrected"
                if result['gt_clusters']:
                    clusters_str = ', '.join(result['gt_clusters'])
                    title = f"{title}\n{clusters_str}"
                ax.set_title(wrap_title(title, max_length=25), fontsize=9)
            ax.axis('off')
        
        # GT Oracle
        if cols_per_image >= 5 and result['gt_oracle'] is not None:
            ax = plt.subplot(rows, cols, row * cols + col_start + 5)
            ax.imshow(result['gt_oracle'])
            if row == 0:
                ax.set_title(wrap_title("GT Mask True Illuminant Corrected", max_length=25), fontsize=9)
            ax.axis('off')
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"batch_{num_images}_images.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------- Processing Functions ----------
def handle_batch_mode(args, model, cluster_mapping):
    """
    Handle batch processing mode.
    
    Args:
        args: Command line arguments
        model: PyTorch model
        cluster_mapping: Dictionary mapping clusters to white points
    """

    if not os.path.exists(LSMI_IMAGES_DIR):
        print(f"Error: Images directory not found: {LSMI_IMAGES_DIR}")
        return
    
    available_images = [f for f in os.listdir(LSMI_IMAGES_DIR) if f.endswith(('.nef', '.NEF', '.tiff', '.TIFF', '.tif', '.TIF'))]
    if len(available_images) == 0:
        print(f"Error: No images found in {LSMI_IMAGES_DIR}")
        return
    
    import random
    num_to_select = min(args.num_images, len(available_images))
    selected_images = random.sample(available_images, num_to_select)
    
    all_results = []
    for img_file in selected_images:
        img_path = os.path.join(LSMI_IMAGES_DIR, img_file)
        result = process_single_image(img_path, model, cluster_mapping, args)
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("Error: No images were successfully processed")
        return
    
    create_batch_visualization(all_results, args)

def load_and_normalize_raw_image(image_path):
    """
    Load and normalize raw image to [0, 1] range.
    
    Args:
        image_path: Path to raw image file
    
    Returns:
        Normalized image array in range [0, 1]
    """

    raw_linear = process_raw_image(image_path, srgb=False)
    
    if raw_linear.dtype == np.uint8:
        raw_linear = raw_linear.astype(np.float32) / 255.0
    elif raw_linear.dtype == np.uint16:
        raw_linear = raw_linear.astype(np.float32) / 65535.0
    else:
        raw_linear = raw_linear.astype(np.float32)
        if raw_linear.max() > 1.0:
            raw_linear = raw_linear / raw_linear.max()
    
    return np.clip(raw_linear, 0.0, 1.0)

def prepare_model_input(image_path):
    """
    Prepare image tensor for model input.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Preprocessed image tensor
    """

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    if image_path.lower().endswith('.nef'):
        rgb_array = process_raw_image(image_path, srgb=False)
        pil_img = Image.fromarray(rgb_array)
    else:
        pil_img = Image.open(image_path).convert('RGB')
    
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def generate_cams(model, args, img_tensor, raw_shape):
    """
    Generate CAMs for all clusters.
    
    Args:
        model: PyTorch model
        args: Command line arguments
        img_tensor: Preprocessed image tensor
        raw_shape: Shape of raw image
    
    Returns:
        Tuple of (pred_class, weights_dict, weights_stack, used_clusters)
    """

    cam = create_cam(model, args.model, args.cam)
    
    with torch.no_grad():
        outputs = model(img_tensor) if args.model != 'confidence' else model(img_tensor)[0]
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = int(outputs.argmax(dim=1)[0].item())
        pred_class = TRAIN_LABELS[pred_idx]
    
    cams_raw = {}
    for i, class_name in enumerate(TRAIN_LABELS):
        targets = [ClassifierOutputTarget(i)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        grayscale_cam = normalize_cam_to_0_1(grayscale_cam)
        cam_resized = cv2.resize(grayscale_cam, (raw_shape[1], raw_shape[0]), interpolation=cv2.INTER_LINEAR)
        cams_raw[class_name] = cam_resized
    
    # Smooth cams
    cams_norm = {}
    for k, v in cams_raw.items():
        n = normalize_cam_to_0_1(v)
        s = smooth_mask(n, ksize=args.smooth_ksize)
        s = normalize_cam_to_0_1(s)
        cams_norm[k] = s
    
    weights_dict, weights_stack = cams_to_softmax_weights(cams_norm, temp=args.temp)
    
    # Determine used clusters
    cluster_max_weights = {name: float(weights_dict[name].max()) for name in weights_dict.keys()}
    used_clusters = {name for name, m in cluster_max_weights.items() if m >= args.threshold}
    
    return pred_class, weights_dict, weights_stack, used_clusters

def compute_corrections(raw_linear, cluster_mapping, weights_stack, pred_class):
    """
    Compute corrected images per cluster and blend them.
    
    Args:
        raw_linear: Linear raw image array
        cluster_mapping: Dictionary mapping clusters to white points
        weights_stack: Stacked weight arrays
        pred_class: Predicted class name
    
    Returns:
        Tuple of (corrected_per_cluster, base_correction, final_image)
    """

    corrected_per_cluster = {}
    for name in TRAIN_LABELS:
        wp = cluster_mapping[name]['wp']
        corrected_per_cluster[name] = apply_correction(raw_linear, wp)
    
    base_correction = corrected_per_cluster[pred_class]
    
    # Blend weighted sum across clusters
    accumulated = np.zeros_like(raw_linear, dtype=np.float32)
    for i, name in enumerate(TRAIN_LABELS):
        w = weights_stack[..., i]
        w3 = w[..., None]
        accumulated += corrected_per_cluster[name] * w3
    
    weight_sum = weights_stack.sum(axis=-1)
    zero_mask = (weight_sum < 1e-6)[..., None]
    final_image = accumulated.copy()
    final_image[zero_mask.squeeze(-1)] = base_correction[zero_mask.squeeze(-1)]
    
    return corrected_per_cluster, base_correction, final_image

def compute_gt_corrections(raw_linear, img_name, corrected_per_cluster, base_correction):
    """
    Compute GT mask corrections if available.
    
    Args:
        raw_linear: Linear raw image array
        img_name: Image name identifier
        corrected_per_cluster: Dictionary of corrected images per cluster
        base_correction: Base correction image
    
    Returns:
        Tuple of (gt_corrected_srgb, gt_clusters, gt_oracle_srgb)
    """

    gt_corrected_srgb = None
    gt_clusters = None
    gt_oracle_srgb = None
    
    mask_path = os.path.join(LSMI_MASKS_DIR, f"{img_name}.npy")
    meta_path = os.path.join(LSMI_TEST_PACKAGE, "meta.json")
    
    if not os.path.exists(mask_path):
        return gt_corrected_srgb, gt_clusters, gt_oracle_srgb
    
    gt_mask = load_mask(mask_path, target_shape=raw_linear.shape[:2])
    
    if not os.path.exists(meta_path):
        print(f"Warning: meta.json not found at {meta_path}, skipping GT correction")
        return gt_corrected_srgb, gt_clusters, gt_oracle_srgb
    
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    if img_name not in meta_data:
        print(f"Warning: {img_name} not found in meta.json, skipping GT correction")
        return gt_corrected_srgb, gt_clusters, gt_oracle_srgb
    
    meta = meta_data[img_name]
    num_lights = meta.get("NumOfLights", 2)
    
    if gt_mask.shape[2] != num_lights:
        print(f"Warning: Mask has {gt_mask.shape[2]} channels but meta.json says {num_lights} lights")
    
    cluster_centers = load_cluster_centers()
    light_clusters = []
    
    for light_num in range(1, num_lights + 1):
        light_key = f"Light{light_num}"
        if light_key not in meta:
            print(f"Warning: {light_key} not found in meta.json")
            continue
        
        light_rgb = np.array(meta[light_key], dtype=np.float32)
        cluster_name, error = map_illuminant_to_cluster(light_rgb, cluster_centers)
        light_clusters.append(cluster_name)
    
    if len(light_clusters) == num_lights and gt_mask.shape[2] >= num_lights:
        # GT cluster correction

        gt_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
        for i, cluster_name in enumerate(light_clusters):
            weight = gt_mask[:, :, i]
            weight_3d = weight[..., None]
            gt_accumulated += corrected_per_cluster[cluster_name] * weight_3d
        
        gt_total_weight = gt_mask.sum(axis=-1, keepdims=True)
        gt_mask_valid = gt_total_weight > 1e-6
        gt_final = np.where(gt_mask_valid, gt_accumulated / (gt_total_weight + 1e-12), base_correction)
        gt_corrected_srgb = lin_to_srgb(gt_final)
        gt_clusters = light_clusters
        
        # GT Oracle correction
        gt_oracle_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
        
        for i in range(num_lights):
            light_key = f"Light{i+1}"
            if light_key not in meta:
                continue
            
            light_rgb = np.array(meta[light_key], dtype=np.float32)
            if light_rgb.sum() > 1e-8:
                light_wp = light_rgb / light_rgb.sum()
            else:
                light_wp = light_rgb
            
            light_corrected = apply_correction(raw_linear, light_wp)
            weight = gt_mask[:, :, i]
            weight_3d = weight[..., None]
            gt_oracle_accumulated += light_corrected * weight_3d
        
        gt_oracle_final = np.where(gt_mask_valid, gt_oracle_accumulated / (gt_total_weight + 1e-12), base_correction)
        gt_oracle_srgb = lin_to_srgb(gt_oracle_final)
    else:
        print(f"Warning: Could not map all {num_lights} lights to clusters or mask shape mismatch")
    
    return gt_corrected_srgb, gt_clusters, gt_oracle_srgb

def create_single_image_visualization(img_name, original_srgb, final_srgb, base_srgb, 
                                     gt_corrected_srgb, gt_clusters, gt_oracle_srgb, args):
    """
    Create and save single image visualization.
    
    Args:
        img_name: Image name identifier
        original_srgb: Original sRGB image
        final_srgb: Final corrected sRGB image
        base_srgb: Base correction sRGB image
        gt_corrected_srgb: Ground truth cluster corrected image
        gt_clusters: List of ground truth cluster names
        gt_oracle_srgb: Ground truth oracle corrected image
        args: Command line arguments
    """

    os.makedirs(args.output, exist_ok=True)
    
    num_imgs = 3
    if gt_corrected_srgb is not None:
        num_imgs = 4
    if gt_oracle_srgb is not None:
        num_imgs = 5
    
    cols = num_imgs
    rows = 1
    fig = plt.figure(figsize=(4*cols, 4*rows))
    
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(original_srgb)
    ax.set_title("Original")
    ax.axis('off')
    
    ax = plt.subplot(rows, cols, 2)
    ax.imshow(final_srgb)
    ax.set_title("CAM Corrected")
    ax.axis('off')
    
    ax = plt.subplot(rows, cols, 3)
    ax.imshow(base_srgb)
    ax.set_title("Single Illuminant Assumption")
    ax.axis('off')
    
    if gt_corrected_srgb is not None:
        ax = plt.subplot(rows, cols, 4)
        ax.imshow(gt_corrected_srgb)
        if gt_clusters is not None:
            clusters_str = ", ".join(gt_clusters)
            ax.set_title(f"GT Mask Cluster Corrected\n{clusters_str}")
        else:
            ax.set_title("GT Mask Cluster Corrected")
        ax.axis('off')
    
    if gt_oracle_srgb is not None:
        ax = plt.subplot(rows, cols, 5)
        ax.imshow(gt_oracle_srgb)
        ax.set_title("GT Mask True Illuminant Corrected")
        ax.axis('off')
    
    correction_suffix = "wb_only"
    out_path = os.path.join(args.output, f"{img_name}_corrected_{correction_suffix}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    """
    Main function for CAM-guided color correction.
    """
    
    parser = argparse.ArgumentParser(description='Apply CAM-guided Color Correction (improved)')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (NEF or TIFF). Required if --num-images is not set.')
    parser.add_argument('--model', type=str, default='standard', choices=MODEL_CHOICES, help='Model type')
    parser.add_argument('--cam', type=str, default='gradcam', choices=CAM_CHOICES, help='CAM method')
    parser.add_argument('--layer', type=str, default='conv5', help='Target layer name')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'visualizations', 'cam_correction'), help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.10, help='Per-pixel threshold on softmax weight to consider a cluster "used" (default: 0.10)')
    parser.add_argument('--smooth_ksize', type=int, default=41, help='Gaussian blur kernel size for smoothing CAM masks (odd)')
    parser.add_argument('--temp', type=float, default=0.7, help='Softmax temperature (lower -> sharper selection).')
    parser.add_argument('--num-images', type=int, default=None, help='Number of random images to process and display in a grid (rounds down if odd). If set, --image is ignored and images are randomly selected from LSMI_IMAGES_DIR')
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_images is None and args.image is None:
        parser.error("Either --image or --num-images must be provided")

    # Round down if odd
    if args.num_images is not None and args.num_images % 2 == 1:
        args.num_images = args.num_images - 1

    # 1) Load resources
    cluster_mapping = get_wps_for_clusters()

    # 2) Load model
    model = load_model(args.model)

    # Handle batch mode
    if args.num_images is not None and args.num_images > 0:
        handle_batch_mode(args, model, cluster_mapping)
        return

    # Single image mode
    raw_linear = load_and_normalize_raw_image(args.image)
    img_tensor = prepare_model_input(args.image)
    
    pred_class, weights_dict, weights_stack, used_clusters = generate_cams(model, args, img_tensor, raw_linear.shape)
    corrected_per_cluster, base_correction, final_image = compute_corrections(raw_linear, cluster_mapping, weights_stack, pred_class)
    
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    gt_corrected_srgb, gt_clusters, gt_oracle_srgb = compute_gt_corrections(raw_linear, img_name, corrected_per_cluster, base_correction)
    
    final_srgb = lin_to_srgb(final_image)
    base_srgb = lin_to_srgb(base_correction)
    original_srgb = process_raw_image(args.image, srgb=True).astype(np.float32) / 255.0
    
    create_single_image_visualization(img_name, original_srgb, final_srgb, base_srgb, 
                                     gt_corrected_srgb, gt_clusters, gt_oracle_srgb, args)

if __name__ == "__main__":
    main()

