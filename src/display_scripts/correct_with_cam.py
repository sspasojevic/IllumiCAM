#!/usr/bin/env python3
"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

CAM-Guided Spatial White Balance Correction Tool

Performs spatially-aware white balance correction guided by Class Activation Maps.
Processes RAW NEF images using model predictions to identify illuminant regions,
then applies appropriate white balance to each region. Uses softmax-weighted
blending when multiple illuminants are detected.

Outputs visualization showing original, corrected, and CAM heatmaps.

Uses:
    - config.config for paths and model settings
    - src.utils for model loading, CAM generation, and RAW processing
    - Cluster white points for reference
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt display errors
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.io import loadmat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.config import MEAN, STD, DEVICE, IMG_SIZE, NIKON_CCM_MAT, VISUALIZATIONS_DIR, MODEL_PATHS, LSMI_MASKS_DIR
from src.utils import load_model, create_cam, process_raw_image, load_mask, CLUSTER_NAMES
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_CHOICES = list(MODEL_PATHS.keys())
CAM_CHOICES = ['gradcam', 'gradcam++', 'scorecam']

TRAIN_LABELS = CLUSTER_NAMES
print(f"Training Label Order (Model Output): {TRAIN_LABELS}")






def load_cluster_centers():
    """Load cluster centers dict from npy file."""
    path = os.path.join(PROJECT_ROOT, "cluster_centers.npy")
    centers = np.load(path, allow_pickle=True).item()
    return centers

def get_wps_for_clusters():
    """Get white points for each cluster."""
    centers = load_cluster_centers()
    cluster_wps = {}

    print("\nLoading Cluster White Points:")
    for name in TRAIN_LABELS:
        wp = centers[name]
        wp_norm = wp / (wp.sum() + 1e-12)
        cluster_wps[name] = {
            'wp': wp_norm
        }
        print(f"  {name} -> WP: [{wp_norm[0]:.3f}, {wp_norm[1]:.3f}, {wp_norm[2]:.3f}]")
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
        wb = wb / (pmax + 1e-12)
        wb = np.clip(wb, 0.0, 1.0)

    return wb
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

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Apply CAM-guided White Balance Correction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image (NEF or TIFF)')
    parser.add_argument('--model', type=str, default='standard', choices=MODEL_CHOICES, help='Model type')
    parser.add_argument('--cam', type=str, default='gradcam', choices=CAM_CHOICES, help='CAM method')
    parser.add_argument('--layer', type=str, default='conv5', help='Target layer name')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'visualizations', 'cam_correction'), help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.10, help='Per-pixel threshold on softmax weight to consider a cluster "used" (default: 0.10)')
    parser.add_argument('--smooth_ksize', type=int, default=41, help='Gaussian blur kernel size for smoothing CAM masks (odd)')
    parser.add_argument('--temp', type=float, default=0.7, help='Softmax temperature (lower -> sharper selection).')
    parser.add_argument('--debug', action='store_true', help='Save extra debug images')
    args = parser.parse_args()

    # 1) Load resources
    print("Loading resources...")
    cluster_mapping = get_wps_for_clusters()

    # 2) Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

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
    # compute a per-cluster max weight to decide which clusters to render in bottom row
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
    gt_illum_corrected_srgb = None  # Oracle correction using illumination map
    gt_mask_vis = None  # For visualization
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    mask_path = os.path.join(LSMI_MASKS_DIR, f"{img_name}.npy")
    meta_path = os.path.join(os.path.dirname(LSMI_MASKS_DIR), 'meta.json')
    
    if os.path.exists(mask_path) and os.path.exists(meta_path):
        print("Loading GT mask for comparison...")
        try:
            import json
            # Load metadata to get illuminant information
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
           
           # Get scene metadata
            if img_name not in metadata:
                print(f"  No metadata found for {img_name}")
            else:
                scene_meta = metadata[img_name]
                num_lights = scene_meta.get('NumOfLights', 0)
                
                if num_lights == 0:
                    print(f"  Scene has no lights")
                else:
                    # Load mixture mask
                    mixture_mask = np.load(mask_path)
                    if mixture_mask.shape[:2] != raw_linear.shape[:2]:
                        h, w = raw_linear.shape[:2]
                        mixture_mask = cv2.resize(mixture_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Get illuminants and map to clusters
                    illum_wps = []
                    for i in range(1, num_lights + 1):
                        light_key = f'Light{i}'
                        if light_key in scene_meta:
                            chrom = scene_meta[light_key]  # This is already a list [r/g, 1, b/g]
                            # Convert chromaticity to white point  (r/g, g/g=1, b/g) -> (r, g, b)
                            wp = np.array(chrom, dtype=np.float32)
                            wp = wp / (wp.sum() + 1e-12)
                            illum_wps.append(wp)
                    
                    if len(illum_wps) == num_lights:
                        # Map each illuminant WP to nearest cluster
                        illum_clusters = []
                        for wp in illum_wps:
                            # Find nearest cluster
                            min_dist = float('inf')
                            best_cluster = None
                            for name in TRAIN_LABELS:
                                cluster_wp = cluster_mapping[name]['wp']
                                dist = np.linalg.norm(wp - cluster_wp)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cluster = name
                            illum_clusters.append(best_cluster)
                        
                        print(f"  Mapped {num_lights} illuminants to clusters: {illum_clusters}")
                        
                        # Apply GT mixture-based correction
                        gt_accumulated = np.zeros_like(raw_linear, dtype=np.float32)
                        for i, cluster_name in enumerate(illum_clusters):
                            if i < mixture_mask.shape[2]:
                                weight = mixture_mask[:, :, i]
                                weight_3d = weight[..., None]
                                gt_accumulated += corrected_per_cluster[cluster_name] * weight_3d
                        
                        # Normalize by total weight
                        gt_total_weight = mixture_mask.sum(axis=-1, keepdims=True)
                        gt_mask_valid = gt_total_weight > 1e-6
                        gt_final = np.where(gt_mask_valid, gt_accumulated / (gt_total_weight + 1e-12), base_correction)
                        gt_corrected_srgb = lin_to_srgb(gt_final)
                        
                        # Create visualization of GT mask (sum across channels for heatmap)
                        if mixture_mask.shape[2] > 0:
                            gt_mask_vis = mixture_mask[:, :, 0]  # Show first channel
                        
                        print("  GT mask correction computed")
                        
                        # Create GT illumination map correction (oracle)
                        # This creates a per-pixel RGB illuminant map and divides by it directly
                        # Following LSMI paper: ℓ_ab = α*ℓ_a + (1-α)*ℓ_b
                        print("  Computing GT illumination map correction (oracle)...")
                        illum_map = np.zeros_like(raw_linear, dtype=np.float32)
                        
                        for i in range(num_lights):
                            if i < mixture_mask.shape[2]:
                                # Get the illuminant chromaticity [r/g, g/g=1, b/g]
                                light_chrom = illum_wps[i]  # This is already the chromaticity
                                
                                # Weight by mixture mask (α for light 0, (1-α) for light 1, etc.)
                                weight = mixture_mask[:, :, i]
                                weight_3d = weight[..., None]
                                
                                # Add weighted illuminant color to the map
                                # ℓ_ab = α*ℓ_a + (1-α)*ℓ_b
                                illum_map += light_chrom[None, None, :] * weight_3d
                        
                        # The mixture weights already sum to 1, so illum_map is the final per-pixel illuminant
                        # Apply white balance by dividing by illumination map
                        gt_illum_corrected = raw_linear / (illum_map + 1e-12)
                        gt_illum_corrected = np.clip(gt_illum_corrected, 0.0, None)
                        
                        # Scale to [0,1] using percentile
                        pmax = np.percentile(gt_illum_corrected, 99.5)
                        if pmax > 0:
                            gt_illum_corrected = gt_illum_corrected / (pmax + 1e-12)
                            gt_illum_corrected = np.clip(gt_illum_corrected, 0.0, 1.0)
                        
                        gt_illum_corrected_srgb = lin_to_srgb(gt_illum_corrected)
                        print("  GT illumination map correction computed")
        except Exception as e:
            print(f"Could not load GT mask: {e}")
            import traceback
            traceback.print_exc()
    
    # 11) Convert to sRGB for visualization
    final_srgb = lin_to_srgb(final_image)
    base_srgb = lin_to_srgb(base_correction)  # Global correction based on predicted class
    
    # Load original image as sRGB using camera white balance
    original_srgb = process_raw_image(args.image, srgb=True).astype(np.float32) / 255.0

    # 12) Save visualization
    print("Saving visualization...")
    os.makedirs(args.output, exist_ok=True)

    # Create figure with original and corrected images
    if gt_corrected_srgb is not None and gt_illum_corrected_srgb is not None:
        # Show 2x2 grid: original, CAM, GT cluster, GT illum map
        fig = plt.figure(figsize=(20, 20))
        
        # Top left: Original (sRGB)
        ax = plt.subplot(2, 2, 1)
        ax.imshow(original_srgb)
        ax.set_title(f"Original (sRGB)", fontsize=18)
        ax.axis('off')
        
        # Top right: CAM corrected
        ax = plt.subplot(2, 2, 2)
        ax.imshow(final_srgb)
        ax.set_title(f"CAM White Balance Corrected", fontsize=18)
        ax.axis('off')
        
        # Bottom left: GT cluster corrected
        ax = plt.subplot(2, 2, 3)
        ax.imshow(gt_corrected_srgb)
        ax.set_title(f"GT Cluster Corrected", fontsize=18)
        ax.axis('off')
        
        # Bottom right: GT illumination map corrected (oracle)
        ax = plt.subplot(2, 2, 4)
        ax.imshow(gt_illum_corrected_srgb)
        ax.set_title(f"GT Illumination Map (Oracle)", fontsize=18)
        ax.axis('off')
    elif gt_corrected_srgb is not None:
        # Show 1x2 grid: original, CAM corrected, GT corrected
        fig = plt.figure(figsize=(30, 10))
        
        # Left: Original (sRGB)
        ax = plt.subplot(1, 3, 1)
        ax.imshow(original_srgb)
        ax.set_title(f"Original (sRGB)", fontsize=18)
        ax.axis('off')
        
        # Middle: CAM corrected
        ax = plt.subplot(1, 3, 2)
        ax.imshow(final_srgb)
        ax.set_title(f"CAM White Balance Corrected", fontsize=18)
        ax.axis('off')
        
        # Right: GT corrected
        ax = plt.subplot(1, 3, 3)
        ax.imshow(gt_corrected_srgb)
        ax.set_title(f"GT Mask White Balance Corrected", fontsize=18)
        ax.axis('off')
    else:
        # Show 1x3 grid: original, global correction, CAM corrected
        fig = plt.figure(figsize=(30, 10))
        
        # Left: Original (sRGB)
        ax = plt.subplot(1, 3, 1)
        ax.imshow(original_srgb)
        ax.set_title(f"Original (sRGB)", fontsize=18)
        ax.axis('off')
        
        # Middle: Global illumination correction
        ax = plt.subplot(1, 3, 2)
        ax.imshow(base_srgb)
        ax.set_title(f"Global Correction ({pred_class})", fontsize=18)
        ax.axis('off')
        
        # Right: CAM corrected
        ax = plt.subplot(1, 3, 3)
        ax.imshow(final_srgb)
        ax.set_title(f"CAM White Balance Corrected", fontsize=18)
        ax.axis('off')

    correction_suffix = "wb_only"
    out_path = os.path.join(args.output, f"{img_name}_corrected_{correction_suffix}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
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
