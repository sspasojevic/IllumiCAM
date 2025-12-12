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

Outputs visualization showing original, corrected, and CAM heatmaps.

Uses:
    - config.config for paths and model settings
    - src.utils for model loading, CAM generation, and RAW processing
    - Nikon D810 reference white points from LSMI dataset
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.io import loadmat
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.config import MEAN, STD, DEVICE, IMG_SIZE, NIKON_CCM_MAT, VISUALIZATIONS_DIR, MODEL_PATHS, LSMI_MASKS_DIR, LSMI_TEST_PACKAGE, LSMI_IMAGES_DIR
from src.utils import load_model, create_cam, process_raw_image, load_mask, CLUSTER_NAMES
from src.data_loader import get_datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_CHOICES = list(MODEL_PATHS.keys())
CAM_CHOICES = ['gradcam', 'gradcam++', 'scorecam']

_, _, _, TRAIN_LABELS = get_datasets()
print(f"Training Label Order (Model Output): {TRAIN_LABELS}")

MAT_PATH = NIKON_CCM_MAT
def load_nikon_wps_fixed():
    """Load Nikon reference white points with robust parsing."""
    print(f"Loading Nikon reference white points from: {MAT_PATH}")
    mat = loadmat(MAT_PATH)
    entries = mat['wps_ccms'].reshape(-1)

    illum_names = []
    illum_wps3 = []

    for e in entries:
        name = str(e['name'][0])
        wp2 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)

        if wp2.shape[0] == 2:
            r_g, b_g = wp2
            wp3 = np.array([r_g, 1.0, b_g], dtype=np.float32)
            wp3 = wp3 / wp3.sum()
        else:
            wp3 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)
            wp3 = wp3 / (wp3.sum() + 1e-12)

        illum_names.append(name)
        illum_wps3.append(wp3)

    illum_wps3 = np.stack(illum_wps3, axis=0)

    print(f"Loaded {len(illum_names)} reference illuminants (Fixed Parsing)")
    return illum_wps3, illum_names

def choose_nikon_wp_3d(wp_rgb, illuminant_wps, names):
    """Find closest Nikon reference white point (euclidean in chromaticity)."""
    distances = np.linalg.norm(illuminant_wps - wp_rgb[None, :], axis=1)
    idx = int(np.argmin(distances))
    return illuminant_wps[idx], names[idx], distances[idx]

def angular_error(x, y):
    """Calculate angular error between two RGB vectors."""
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
    """Map an illuminant RGB to the closest cluster using angular error."""
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
    """Load cluster centers dict from npy file."""
    path = os.path.join(PROJECT_ROOT, "cluster_centers.npy")
    centers = np.load(path, allow_pickle=True).item()
    return centers

def get_wps_for_clusters(illum_wps3, illum_names):
    """Map each cluster to nearest Nikon white point."""
    centers = load_cluster_centers()
    cluster_wps = {}

    print("\nMapping Clusters to Nikon White Points:")
    for name in TRAIN_LABELS:
        wp = centers[name]
        wp_norm = wp / (wp.sum() + 1e-12)
        ref_wp, ref_name, dist = choose_nikon_wp_3d(wp_norm, illum_wps3, illum_names)
        cluster_wps[name] = {
            'wp': wp_norm,
            'ref_name': ref_name,
            'dist': float(dist)
        }
        print(f"  {name} -> {ref_name} (dist={dist:.4f})")
    return cluster_wps

# ---------- Raw correction ----------
def apply_correction(raw_img, wp):
    """
    Apply white balance to linear raw image.
    
    Args:
        raw_img: Linear raw image
        wp: White point (3-vector)
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
    """Scale CAM to [0,1] by min/max."""
    mn, mx = float(cam.min()), float(cam.max())
    if mx - mn < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - mn) / (mx - mn)).astype(np.float32)

def smooth_mask(mask, ksize=41, sigma=0):
    """Smooth mask with Gaussian blur."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma)

def cams_to_softmax_weights(cam_dict, eps=1e-8, temp=1.0):
    """Convert CAM dict to per-pixel softmax weights."""
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
    linear_rgb: array in range [0, 1]
    returns: sRGB array in range [0, 1]
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
    """
    return np.clip(x, 0.0, 1.0)

# ---------- Processing Functions ----------
def process_single_image(image_path, model, illum_wps3, illum_names, cluster_mapping, args):
    """Process a single image and return visualization data."""
    try:
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
            try:
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
            except Exception as e:
                pass  # Silently skip GT if not available
        
        # Convert to sRGB for visualization
        final_srgb = lin_to_srgb(final_image)
        base_srgb = lin_to_srgb(base_correction)
        original_vis = view_as_linear(raw_linear)
        
        return {
            'img_name': img_name,
            'original': original_vis,
            'cam_corrected': final_srgb,
            'base': base_srgb,
            'gt_cluster': gt_corrected_srgb,
            'gt_oracle': gt_oracle_srgb,
            'gt_clusters': gt_clusters
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_batch_visualization(all_results, args):
    """Create a grid visualization for multiple images."""
    num_images = len(all_results)
    
    # Determine number of columns per image (3-5 depending on GT availability)
    # Check if any image has GT data
    has_gt_cluster = any(r['gt_cluster'] is not None for r in all_results)
    has_gt_oracle = any(r['gt_oracle'] is not None for r in all_results)
    
    cols_per_image = 3  # original, cam corrected, base
    if has_gt_cluster:
        cols_per_image = 4
    if has_gt_oracle:
        cols_per_image = 5
    
    # Arrange in grid: each image takes cols_per_image columns
    # For 8 images with 4 cols each: arrange as 2 rows x 4 images = 8 images
    # Each row has 4 images * 4 cols = 16 columns total
    if num_images >= 2:
        images_per_row = num_images // 2  # 2 rows
    else:
        images_per_row = 1
    
    num_rows = (num_images + images_per_row - 1) // images_per_row
    cols = images_per_row * cols_per_image
    rows = num_rows
    
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    
    for img_idx, result in enumerate(all_results):
        row = img_idx // images_per_row
        col_start = (img_idx % images_per_row) * cols_per_image
        
        # Original
        ax = plt.subplot(rows, cols, row * cols + col_start + 1)
        ax.imshow(result['original'])
        if row == 0 and img_idx == 0:
            ax.set_title("Original (Raw Linear)")
        ax.axis('off')
        
        # CAM Corrected
        ax = plt.subplot(rows, cols, row * cols + col_start + 2)
        ax.imshow(result['cam_corrected'])
        if row == 0 and img_idx == 0:
            ax.set_title("CAM Corrected")
        ax.axis('off')
        
        # Base
        ax = plt.subplot(rows, cols, row * cols + col_start + 3)
        ax.imshow(result['base'])
        if row == 0 and img_idx == 0:
            ax.set_title("Single Illuminant Assumption")
        ax.axis('off')
        
        # GT Cluster (if available)
        if cols_per_image >= 4 and result['gt_cluster'] is not None:
            ax = plt.subplot(rows, cols, row * cols + col_start + 4)
            ax.imshow(result['gt_cluster'])
            if row == 0 and img_idx == 0:
                title = "GT Mask Cluster Corrected"
                if result['gt_clusters']:
                    title += f"\n{', '.join(result['gt_clusters'])}"
                ax.set_title(title)
            ax.axis('off')
        
        # GT Oracle (if available)
        if cols_per_image >= 5 and result['gt_oracle'] is not None:
            ax = plt.subplot(rows, cols, row * cols + col_start + 5)
            ax.imshow(result['gt_oracle'])
            if row == 0 and img_idx == 0:
                ax.set_title("GT Mask True Illuminant Corrected")
            ax.axis('off')
        
        # Add image name as label
        if row == 0:
            fig.text((col_start + cols_per_image / 2) / cols, 1.0 - (row + 0.95) / rows, 
                    result['img_name'], ha='center', va='top', fontsize=8)
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"batch_{num_images}_images.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved batch visualization: {out_path}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Apply CAM-guided Color Correction (improved)')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (NEF or TIFF). Required if --num-images is not set.')
    parser.add_argument('--model', type=str, default='standard', choices=MODEL_CHOICES, help='Model type')
    parser.add_argument('--cam', type=str, default='gradcam', choices=CAM_CHOICES, help='CAM method')
    parser.add_argument('--layer', type=str, default='conv5', help='Target layer name')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'visualizations', 'cam_correction'), help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.10, help='Per-pixel threshold on softmax weight to consider a cluster "used" (default: 0.10)')
    parser.add_argument('--smooth_ksize', type=int, default=41, help='Gaussian blur kernel size for smoothing CAM masks (odd)')
    parser.add_argument('--temp', type=float, default=0.7, help='Softmax temperature (lower -> sharper selection).')
    parser.add_argument('--debug', action='store_true', help='Save extra debug images')
    parser.add_argument('--num-images', type=int, default=None, help='Number of random images to process and display in a grid (rounds down if odd). If set, --image is ignored and images are randomly selected from LSMI_IMAGES_DIR')
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_images is None and args.image is None:
        parser.error("Either --image or --num-images must be provided")

    # Round down if odd
    if args.num_images is not None and args.num_images % 2 == 1:
        args.num_images = args.num_images - 1
        print(f"Rounded down to {args.num_images} images (must be even)")

    # 1) Load resources
    print("Loading resources...")
    illum_wps3, illum_names = load_nikon_wps_fixed()
    cluster_mapping = get_wps_for_clusters(illum_wps3, illum_names)

    # 2) Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    # Handle batch mode
    if args.num_images is not None and args.num_images > 0:
        # Get list of available images
        if not os.path.exists(LSMI_IMAGES_DIR):
            print(f"Error: Images directory not found: {LSMI_IMAGES_DIR}")
            return
        
        available_images = [f for f in os.listdir(LSMI_IMAGES_DIR) if f.endswith(('.nef', '.NEF', '.tiff', '.TIFF', '.tif', '.TIF'))]
        if len(available_images) == 0:
            print(f"Error: No images found in {LSMI_IMAGES_DIR}")
            return
        
        # Randomly select N images
        import random
        random.seed(42)  # For reproducibility
        num_to_select = min(args.num_images, len(available_images))
        selected_images = random.sample(available_images, num_to_select)
        print(f"Selected {num_to_select} images: {[os.path.splitext(f)[0] for f in selected_images]}")
        
        # Process each image
        all_results = []
        for img_file in selected_images:
            img_path = os.path.join(LSMI_IMAGES_DIR, img_file)
            print(f"\nProcessing: {img_file}")
            result = process_single_image(img_path, model, illum_wps3, illum_names, cluster_mapping, args)
            if result is not None:
                all_results.append(result)
        
        if len(all_results) == 0:
            print("Error: No images were successfully processed")
            return
        
        # Create grid visualization
        create_batch_visualization(all_results, args)
        return

    # Single image mode
    # 3) Load raw linear image for correction
    print(f"Loading image: {args.image}")

    # We want to preserve the original brightness (darkness) relative to full well/saturation.
    raw_linear = process_raw_image(args.image, srgb=False)
    
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

    # 4) Prepare model input (must match visualize_cam processing)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Load image for model input (consistent with visualize_cam.py)
    if args.image.lower().endswith('.nef'):
        rgb_array = process_raw_image(args.image, srgb=False)
        pil_img = Image.fromarray(rgb_array)
    else:
        pil_img = Image.open(args.image).convert('RGB')

    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 5) Get CAMs
    print("Generating CAMs...")
    cam = create_cam(model, args.model, args.cam)

    # Get logits/probs and predicted class
    with torch.no_grad():
        outputs = model(img_tensor) if args.model != 'confidence' else model(img_tensor)[0]
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = int(outputs.argmax(dim=1)[0].item())
        pred_class = TRAIN_LABELS[pred_idx]

    print(f"Predicted class by model: {pred_class} ({probs[pred_idx]:.2%})")

    # Generate CAM for each cluster name present in cluster_mapping and TRAIN_LABELS
    cams_raw = {}
    for i, class_name in enumerate(TRAIN_LABELS):
        targets = [ClassifierOutputTarget(i)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]  # e.g., 224x224
        # normalize to 0..1
        grayscale_cam = normalize_cam_to_0_1(grayscale_cam)
        # Resize to original raw size
        cam_resized = cv2.resize(grayscale_cam, (raw_linear.shape[1], raw_linear.shape[0]), interpolation=cv2.INTER_LINEAR)
        cams_raw[class_name] = cam_resized

    # 6) Smooth cams and prepare per-cluster masks
    cams_norm = {}
    for k, v in cams_raw.items():
        # Normalize again (robust), then smooth
        n = normalize_cam_to_0_1(v)
        s = smooth_mask(n, ksize=args.smooth_ksize)
        s = normalize_cam_to_0_1(s)
        cams_norm[k] = s

    # 7) Convert cams -> per-pixel softmax weights (competing)
    weights_dict, weights_stack = cams_to_softmax_weights(cams_norm, temp=args.temp)
    # weights_stack shape: H x W x C

    # Determine which clusters are used by thresholding their softmax weight at each pixel
    used_clusters = set()
    H, W = raw_linear.shape[:2]
    # compute a per-cluster max weight to determine which clusters are active
    cluster_max_weights = {name: float(weights_dict[name].max()) for name in weights_dict.keys()}
    for name, m in cluster_max_weights.items():
        if m >= args.threshold:
            used_clusters.add(name)

    print("Clusters used (max softmax weight >= threshold):", used_clusters)

    # 8) Precompute corrected images per cluster
    corrected_per_cluster = {}
    for name in TRAIN_LABELS:
        wp = cluster_mapping[name]['wp']
        corrected_per_cluster[name] = apply_correction(raw_linear, wp)

    # Compute base correction (predicted class)
    base_correction = corrected_per_cluster[pred_class]

    # 9) Blend: weighted sum across clusters (weights sum to 1 per-pixel)
    # Weighted composition
    accumulated = np.zeros_like(raw_linear, dtype=np.float32)
    for i, name in enumerate(TRAIN_LABELS):
        w = weights_stack[..., i]  # HxW
        w3 = w[..., None]  # HxWx1
        accumulated += corrected_per_cluster[name] * w3

    # If for some pixels the mask numerically sums to zero (shouldn't happen), use base_correction
    weight_sum = weights_stack.sum(axis=-1)
    zero_mask = (weight_sum < 1e-6)[..., None]
    final_image = accumulated.copy()
    final_image[zero_mask.squeeze(-1)] = base_correction[zero_mask.squeeze(-1)]

    # 10) Compute GT mask correction if available
    gt_corrected_srgb = None
    gt_clusters = None  # Store cluster names for title
    gt_oracle_srgb = None  # Oracle correction using GT illuminant RGBs directly
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    mask_path = os.path.join(LSMI_MASKS_DIR, f"{img_name}.npy")
    meta_path = os.path.join(LSMI_TEST_PACKAGE, "meta.json")
    
    if os.path.exists(mask_path):
        print("Loading GT mask for comparison...")
        try:
            gt_mask = load_mask(mask_path, target_shape=raw_linear.shape[:2])
            print(f"GT mask shape: {gt_mask.shape}")
            
            # Load meta.json to get illuminant RGBs and map to clusters
            if not os.path.exists(meta_path):
                print(f"Warning: meta.json not found at {meta_path}, skipping GT correction")
            else:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                if img_name not in meta_data:
                    print(f"Warning: {img_name} not found in meta.json, skipping GT correction")
                else:
                    meta = meta_data[img_name]
                    num_lights = meta.get("NumOfLights", 2)
                    
                    # Verify mask has correct number of channels
                    if gt_mask.shape[2] != num_lights:
                        print(f"Warning: Mask has {gt_mask.shape[2]} channels but meta.json says {num_lights} lights")
                    
                    # Load cluster centers for mapping
                    cluster_centers = load_cluster_centers()
                    
                    # Map each light to a cluster
                    light_clusters = []
                    
                    for light_num in range(1, num_lights + 1):
                        light_key = f"Light{light_num}"
                        if light_key not in meta:
                            print(f"Warning: {light_key} not found in meta.json")
                            continue
                        
                        light_rgb = np.array(meta[light_key], dtype=np.float32)
                        cluster_name, error = map_illuminant_to_cluster(light_rgb, cluster_centers)
                        light_clusters.append(cluster_name)
                        print(f"  {light_key} RGB {light_rgb} -> Cluster {cluster_name} (error: {error:.2f}°)")
                    
                    if len(light_clusters) == num_lights and gt_mask.shape[2] >= num_lights:
                        # Blend using GT mask weights (cluster-based)
                        gt_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
                        
                        for i, cluster_name in enumerate(light_clusters):
                            # Channel i in mask corresponds to Light{i+1}
                            weight = gt_mask[:, :, i]
                            weight_3d = weight[..., None]
                            gt_accumulated += corrected_per_cluster[cluster_name] * weight_3d
                        
                        # Normalize by total weight
                        gt_total_weight = gt_mask.sum(axis=-1, keepdims=True)
                        gt_mask_valid = gt_total_weight > 1e-6
                        gt_final = np.where(gt_mask_valid, gt_accumulated / (gt_total_weight + 1e-12), base_correction)
                        gt_corrected_srgb = lin_to_srgb(gt_final)
                        gt_clusters = light_clusters  # Store for title
                        print(f"GT mask correction computed ({num_lights} lights)")
                        
                        # Oracle correction: use GT illuminant RGBs directly (no cluster mapping)
                        print("Computing GT Oracle correction (using illuminant RGBs directly)...")
                        gt_oracle_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
                        
                        for i in range(num_lights):
                            light_key = f"Light{i+1}"
                            if light_key not in meta:
                                continue
                            
                            # Get GT illuminant RGB
                            light_rgb = np.array(meta[light_key], dtype=np.float32)
                            # Normalize to chromaticity (sum=1)
                            if light_rgb.sum() > 1e-8:
                                light_wp = light_rgb / light_rgb.sum()
                            else:
                                light_wp = light_rgb
                            
                            # Find closest Nikon white point for this illuminant
                            ref_wp, ref_name, wp_dist = choose_nikon_wp_3d(light_wp, illum_wps3, illum_names)
                            
                            # Apply correction using GT illuminant WP directly
                            light_corrected = apply_correction(raw_linear, light_wp)
                            
                            # Blend using mask weights
                            weight = gt_mask[:, :, i]
                            weight_3d = weight[..., None]
                            gt_oracle_accumulated += light_corrected * weight_3d
                            
                            print(f"  {light_key} RGB {light_rgb} -> WP {ref_name} (dist={wp_dist:.4f})")
                        
                        # Normalize by total weight
                        gt_oracle_final = np.where(gt_mask_valid, gt_oracle_accumulated / (gt_total_weight + 1e-12), base_correction)
                        gt_oracle_srgb = lin_to_srgb(gt_oracle_final)
                        print("GT Oracle correction computed")
                    else:
                        print(f"Warning: Could not map all {num_lights} lights to clusters or mask shape mismatch")
                
        except Exception as e:
            print(f"Could not load GT mask: {e}")
            import traceback
            traceback.print_exc()
    
    # 11) Convert to sRGB for visualization
    final_srgb = lin_to_srgb(final_image)
    base_srgb = lin_to_srgb(base_correction)
    original_vis = view_as_linear(raw_linear)

    # 12) Save visualization
    print("Saving visualization...")
    os.makedirs(args.output, exist_ok=True)

    # Prepare figure layout
    # Add extra columns for GT corrections if available
    num_imgs = 3  # original, cam corrected, base
    if gt_corrected_srgb is not None:
        num_imgs = 4  # add GT corrected clusters
    if gt_oracle_srgb is not None:
        num_imgs = 5  # add GT oracle
    
    cols = num_imgs
    rows = 1
    fig = plt.figure(figsize=(4*cols, 4*rows))

    # Top row: Original
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(original_vis)
    ax.set_title("Original (Raw Linear)")
    ax.axis('off')

    # Top row: CAM Corrected
    ax = plt.subplot(rows, cols, 2)
    ax.imshow(final_srgb)
    ax.set_title("CAM Corrected")
    ax.axis('off')

    # Top row: Base Correction
    ax = plt.subplot(rows, cols, 3)
    ax.imshow(base_srgb)
    ax.set_title("Single Illuminant Assumption")
    ax.axis('off')
    
    # Top row: GT Corrected Clusters (if available)
    if gt_corrected_srgb is not None:
        ax = plt.subplot(rows, cols, 4)
        ax.imshow(gt_corrected_srgb)
        if gt_clusters is not None:
            clusters_str = ", ".join(gt_clusters)
            ax.set_title(f"GT Mask Cluster Corrected\n{clusters_str}")
        else:
            ax.set_title("GT Mask Cluster Corrected")
        ax.axis('off')
    
    # GT Oracle (if available)
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
    print(f"Saved visualization: {out_path}")

    # Optionally save debug images
    if args.debug:
        debug_dir = os.path.join(args.output, f"{img_name}_debug")
        os.makedirs(debug_dir, exist_ok=True)
        # save per-cluster corrected images and weight maps
        for name, corr in corrected_per_cluster.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}_corrected.png"), (lin_to_srgb(corr) * 255).astype(np.uint8)[:, :, ::-1])
            w = (weights_dict[name] * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f"{name}_weight.png"), w)
        # save final and base
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_final.png"), (final_srgb * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_base.png"), (base_srgb * 255).astype(np.uint8)[:, :, ::-1])
        print(f"Saved debug images to: {debug_dir}")

    # Print summary of what was used
    print("Summary:")
    print(f"  Correction mode: White Balance Only")
    print(f"  Predicted class: {pred_class}")
    print(f"  Clusters with max softmax >= {args.threshold:.2f}: {sorted(list(used_clusters))}")
    print("Done.")

if __name__ == "__main__":
    main()

