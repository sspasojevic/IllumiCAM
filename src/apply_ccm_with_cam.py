#!/usr/bin/env python3
"""
CAM-guided spatial color correction.

Takes a raw image (NEF) from Data/LSMI_Test_Package/images, runs a chosen CAM
method on a chosen model, and wherever the CAM shows activation above a
threshold we apply the corresponding cluster's CCM (and WB) to the linear raw
image. CAMs are normalized and converted to per-pixel softmax weights so
multiple cluster activations compete rather than accumulate.

Outputs:
 - {image_name}_corrected.png : visualization with original / corrected / CAMs
 - prints which cluster CCMs were actually used
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import from existing modules
from src.visualize_cam import (
    MODELS, MODEL_PATHS, CAM_METHODS,
    ModelWrapper, get_available_layers, create_cam_instance,
    MEAN, STD, DEVICE, IMG_SIZE
)
from src.data_loader import get_datasets # Import to get correct label order
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.lsmi_utils import process_raw_image

# We need the model's training label order
_, _, _, TRAIN_LABELS = get_datasets()
print(f"Training Label Order (Model Output): {TRAIN_LABELS}")
# Expected: ['Cool', 'Neutral', 'Very_Cool', 'Very_Warm', 'Warm'] (Alphabetical)

MAT_PATH = os.path.join(PROJECT_ROOT, "Data", "info", "Info", "reference_wps_ccms_nikond810.mat")

# ---------- Nikon CCM helpers (same idea as your original) ----------
def load_nikon_ccms_fixed():
    """Load Nikon reference WPs/CCMs with robust parsing."""
    print(f"Loading Nikon reference WPs/CCMs from: {MAT_PATH}")
    mat = loadmat(MAT_PATH)
    entries = mat['wps_ccms'].reshape(-1)

    illum_names = []
    illum_wps3 = []
    illum_ccms = []

    for e in entries:
        name = str(e['name'][0])
        wp2 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)

        if wp2.shape[0] == 2:
            # Interpret as [R/G, B/G] -> [R, G, B] normalized
            r_g, b_g = wp2
            wp3 = np.array([r_g, 1.0, b_g], dtype=np.float32)
            wp3 = wp3 / wp3.sum()
        else:
            wp3 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)
            # ensure normalized chromaticity
            wp3 = wp3 / (wp3.sum() + 1e-12)

        ccm = np.array(e['ccm'][0], dtype=np.float32).reshape(3, 3)
        illum_names.append(name)
        illum_wps3.append(wp3)
        illum_ccms.append(ccm)

    illum_wps3 = np.stack(illum_wps3, axis=0)
    illum_ccms = np.stack(illum_ccms, axis=0)

    print(f"Loaded {len(illum_names)} reference illuminants (Fixed Parsing)")
    return illum_wps3, illum_ccms, illum_names

def choose_nikon_ccm_3d(wp_rgb, illuminant_wps, ccms, names):
    """Find closest Nikon reference CCM for given white point (euclidean in chromaticity)."""
    distances = np.linalg.norm(illuminant_wps - wp_rgb[None, :], axis=1)
    idx = int(np.argmin(distances))
    return ccms[idx], names[idx], distances[idx]

def load_cluster_centers():
    """Load cluster centers dict from npy file."""
    path = os.path.join(PROJECT_ROOT, "cluster_centers.npy")
    centers = np.load(path, allow_pickle=True).item()
    return centers

def get_ccms_for_clusters(illum_wps3, illum_ccms, illum_names):
    """Map each cluster to nearest Nikon CCM & WP."""
    centers = load_cluster_centers()
    cluster_ccms = {}

    print("\nMapping Clusters to Nikon CCMs:")
    # We map based on the Training Labels to ensure we cover all model outputs
    for name in TRAIN_LABELS:
        wp = centers[name]
        # ensure normalized chromaticity
        wp_norm = wp / (wp.sum() + 1e-12)
        ccm, ref_name, dist = choose_nikon_ccm_3d(wp_norm, illum_wps3, illum_ccms, illum_names)
        cluster_ccms[name] = {
            'ccm': ccm,
            'wp': wp_norm,
            'ref_name': ref_name,
            'dist': float(dist)
        }
        print(f"  {name} -> {ref_name} (dist={dist:.4f})")
    return cluster_ccms

# ---------- Raw correction ----------
def apply_correction(raw_img, wp, ccm):
    """
    Apply WB and CCM to linear raw image.
    raw_img: HxWx3 linear representation in range ~0..1
    wp: 3-vector chromaticity (not necessarily normalized to sum 1, but G ~ 1 assumption not needed here)
    ccm: 3x3 matrix
    returns HxWx3 linear corrected image in range 0..1 (scaled)
    """
    raw = raw_img.astype(np.float32)
    # If raw was integer scaled, assume already normalized before call.
    # Normalize wp so that green ~ 1 (match your earlier approach)
    if wp[1] == 0:
        wp_norm = wp / (wp.sum() + 1e-12)
    else:
        wp_norm = wp / (wp[1] + 1e-12)

    # Avoid divide-by-zero in wb
    wb = raw / (wp_norm[None, None, :] + 1e-12)
    wb = np.clip(wb, 0.0, None)

    # Apply CCM
    h, w, _ = wb.shape
    rendered = wb.reshape(-1, 3) @ ccm.T
    rendered = rendered.reshape(h, w, 3)
    rendered = np.clip(rendered, 0.0, None)

    # Normalize scale to [0,1] by percentile to avoid single bright pixel domination
    pmax = np.percentile(rendered, 99.5)
    if pmax > 0:
        rendered = rendered / (pmax + 1e-12)
        rendered = np.clip(rendered, 0.0, 1.0)

    return rendered

# ---------- CAM normalization / weighting utilities ----------
def normalize_cam_to_0_1(cam):
    """Scale cam to [0,1] by min/max. If constant, returns zeros."""
    mn, mx = float(cam.min()), float(cam.max())
    if mx - mn < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - mn) / (mx - mn)).astype(np.float32)

def smooth_mask(mask, ksize=41, sigma=0):
    """Smooth mask using Gaussian blur (ksize should be odd)."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma)

def cams_to_softmax_weights(cam_dict, eps=1e-8, temp=1.0):
    """
    cam_dict: {name: cam_hxw (float32, assumed normalized 0..1)}
    Returns:
      weights: dict {name: weight_hxw}
      stack_weights: hxw x num_classes (numpy array)
    Approach: exponentiate scaled cams and normalize per-pixel (softmax).
    temp <1 sharpens, >1 smooths.
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
    parser = argparse.ArgumentParser(description='Apply CAM-guided Color Correction (improved)')
    parser.add_argument('--image', type=str, required=True, help='Path to input image (NEF or TIFF)')
    parser.add_argument('--model', type=str, default='standard', choices=list(MODELS.keys()), help='Model type')
    parser.add_argument('--cam', type=str, default='gradcam', choices=list(CAM_METHODS.keys()), help='CAM method')
    parser.add_argument('--layer', type=str, default='conv5', help='Target layer name')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'visualizations', 'cam_correction'), help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.10, help='Per-pixel threshold on softmax weight to consider a cluster "used" (default: 0.10)')
    parser.add_argument('--smooth_ksize', type=int, default=41, help='Gaussian blur kernel size for smoothing CAM masks (odd)')
    parser.add_argument('--temp', type=float, default=0.7, help='Softmax temperature (lower -> sharper selection).')
    parser.add_argument('--debug', action='store_true', help='Save extra debug images')
    args = parser.parse_args()

    # 1) Load resources
    print("Loading resources...")
    illum_wps3, illum_ccms, illum_names = load_nikon_ccms_fixed()
    cluster_mapping = get_ccms_for_clusters(illum_wps3, illum_ccms, illum_names)

    # 2) Load model
    print(f"Loading model: {args.model}")
    model_path = MODEL_PATHS[args.model]
    model_class = MODELS[args.model]
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3) Load raw linear image for correction
    print(f"Loading image: {args.image}")
    # process_raw_image returns numpy array, usually uint8 or uint16 depending on implementation
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
    available_layers = get_available_layers(model, args.model)
    layer_dict = {name: layer for name, layer in available_layers}
    target_layer = layer_dict[args.layer]

    model_wrapper = ModelWrapper(model, args.model)
    cam = create_cam_instance(args.cam, model_wrapper, [target_layer], model, args.model)

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

    # Determine which clusters are "used" by thresholding their softmax weight at each pixel
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
        ccm = cluster_mapping[name]['ccm']
        corrected_per_cluster[name] = apply_correction(raw_linear, wp, ccm)

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

    # 10) Convert to sRGB for visualization
    final_srgb = lin_to_srgb(final_image)
    base_srgb = lin_to_srgb(base_correction)
    # Use view_as_linear for original to keep it dark/green (raw look)
    original_vis = view_as_linear(raw_linear)

    # 11) Save visualization
    print("Saving visualization...")
    os.makedirs(args.output, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(args.image))[0]

    # prepare figure: top row original / corrected / base-correction (for comparison)
    used_list = sorted(list(used_clusters))
    num_cam_plots = max(1, len(used_list))
    cols = max(3, num_cam_plots + 2)  # ensure space for original and corrected
    rows = 2
    fig = plt.figure(figsize=(4*cols, 4*rows))

    ax = plt.subplot(rows, cols, 1)
    ax.imshow(original_vis)
    ax.set_title("Original (Raw Linear)")
    ax.axis('off')

    ax = plt.subplot(rows, cols, 2)
    ax.imshow(final_srgb)
    ax.set_title(f"Final Corrected (base: {pred_class})")
    ax.axis('off')

    ax = plt.subplot(rows, cols, 3)
    ax.imshow(base_srgb)
    ax.set_title("Base Correction (pred/fallback)")
    ax.axis('off')

    # Bottom row: show the used CAM heatmaps and their max weights
    for i, name in enumerate(used_list):
        ax = plt.subplot(rows, cols, cols + i + 1)
        hmap = cams_norm[name]
        ax.imshow(hmap, cmap='jet')
        ax.set_title(f"CAM: {name}\nmaxW={cluster_max_weights[name]:.3f}")
        ax.axis('off')

    out_path = os.path.join(args.output, f"{img_name}_corrected.png")
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
    print(f"  Predicted class: {pred_class}")
    print(f"  Clusters with max softmax >= {args.threshold:.2f}: {sorted(list(used_clusters))}")
    print("Done.")

if __name__ == "__main__":
    main()
