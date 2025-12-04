#!/usr/bin/env python3
"""
Visualize and save illuminant-corrected images from Nikon D810 dataset.

Usage:
    # Single image preview (saves to visualizations/)
    python visualize_illuminants.py --image path/to/image.tiff
    
    # Grid of random images (saves to visualizations/)
    python visualize_illuminants.py --grid 15
    
    # Specify custom output directory
    python visualize_illuminants.py --grid 15 --output my_visualizations/
"""

import os
import sys
import argparse
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.io import loadmat
from sklearn.cluster import KMeans
import random

# Configuration
DATA_ROOT = os.path.join("Data", "Nikon_D810")
MAT_PATH = "Data/info/Info/reference_wps_ccms_nikond810.mat"
RANDOM_SEED = 42

def load_dataset():
    """Load Nikon D810 dataset with white point information."""
    data_list = []
    print(f"Loading .wp files from {DATA_ROOT}...")
    
    search_pattern = os.path.join(DATA_ROOT, "**", "*.wp")
    files = glob.glob(search_pattern, recursive=True)
    
    for wp_file in files:
        try:
            with open(wp_file, "r") as f:
                line = f.read().strip()
                values = line.replace("\t", " ").split()
                if len(values) >= 3:
                    r, g, b = float(values[0]), float(values[1]), float(values[2])
                    total = r + g + b
                    if total > 0:
                        folder_name = os.path.basename(os.path.dirname(wp_file))
                        image_path = wp_file.replace('.wp', '.tiff')
                        
                        data_list.append({
                            'mean_r': r/total,
                            'mean_g': g/total,
                            'mean_b': b/total,
                            'split': folder_name,
                            'source_file': os.path.basename(wp_file),
                            'wp_path': wp_file,
                            'image_path': image_path
                        })
        except Exception as e:
            continue
    
    df = pd.DataFrame(data_list)
    if df.empty:
        raise ValueError("No data loaded from dataset!")
    
    print(f"Loaded {len(df)} images")
    return df

def perform_clustering(df):
    """Perform KMeans clustering on chromaticity."""
    X = df[['mean_r', 'mean_g', 'mean_b']].values
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    sorted_clusters = sorted(range(5), key=lambda i: kmeans.cluster_centers_[i][2] / kmeans.cluster_centers_[i][0])
    label_template = ["Very_Warm", "Warm", "Neutral", "Cool", "Very_Cool"]
    cluster_names = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        cluster_names[cluster_id] = label_template[rank] if rank < len(label_template) else f"Cluster_{rank}"
    
    df['cluster_name'] = df['cluster'].map(cluster_names)
    return df, cluster_names, sorted_clusters

def load_nikon_ccms():
    """Load Nikon reference white points and CCMs."""
    print(f"Loading Nikon reference WPs/CCMs from: {MAT_PATH}")
    
    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f"Reference CCM file not found: {MAT_PATH}")
    
    mat = loadmat(MAT_PATH)
    entries = mat['wps_ccms'].reshape(-1)
    
    illum_names = []
    illum_wps3 = []
    illum_ccms = []
    
    for e in entries:
        name = str(e['name'][0])
        wp2 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)
        if wp2.shape[0] == 2:
            r, g = wp2
            b = max(1.0 - r - g, 0.0)
            wp3 = np.array([r, g, b], dtype=np.float32)
        else:
            wp3 = np.array(e['wp'][0], dtype=np.float32).reshape(-1)
        ccm = np.array(e['ccm'][0], dtype=np.float32).reshape(3, 3)
        illum_names.append(name)
        illum_wps3.append(wp3)
        illum_ccms.append(ccm)
    
    illum_wps3 = np.stack(illum_wps3, axis=0)
    illum_ccms = np.stack(illum_ccms, axis=0)
    
    print(f"Loaded {len(illum_names)} reference illuminants")
    return illum_wps3, illum_ccms, illum_names

def choose_nikon_ccm_3d(wp_rgb, illuminant_wps, ccms, names):
    """Find closest Nikon reference CCM for given white point."""
    distances = np.linalg.norm(illuminant_wps - wp_rgb[None, :], axis=1)
    idx = int(np.argmin(distances))
    return ccms[idx], names[idx], distances[idx]

def lin_to_srgb(x):
    """Convert linear RGB to sRGB."""
    x = np.clip(x, 0, None)
    x = x / x.max() if x.max() > 0 else x
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1/2.4) - 0.055,
    )

def process_image(img_path, wp_gt, illum_wps3, illum_ccms, illum_names):
    """Process a single image: load, white balance, apply CCM, convert to sRGB."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load image
    cam_rgb = iio.imread(img_path).astype(np.float32)
    if cam_rgb.max() > 1.0:
        cam_rgb /= cam_rgb.max()
    
    # Raw with illuminant (no WB, no CCM)
    raw_disp = lin_to_srgb(cam_rgb)
    
    # White balance
    wp_norm = wp_gt / max(wp_gt[1], 1e-6)
    wb = cam_rgb / wp_norm[None, None, :]
    wb = np.clip(wb, 0, None)
    
    # Choose and apply CCM
    ref_ccm, ref_name, dist = choose_nikon_ccm_3d(wp_gt, illum_wps3, illum_ccms, illum_names)
    
    h, w, _ = wb.shape
    rendered_flat = wb.reshape(-1, 3) @ ref_ccm.T
    rendered = rendered_flat.reshape(h, w, 3)
    rendered = np.clip(rendered, 0, None)
    rendered = rendered / rendered.max() if rendered.max() > 0 else rendered
    
    # Tone curve
    rendered_disp = lin_to_srgb(rendered)
    
    return raw_disp, rendered_disp, ref_name

def visualize_single_image(img_path, df, illum_wps3, illum_ccms, illum_names, output_dir):
    """Visualize a single image (raw + rendered) and save to file."""
    # Find image in dataset
    img_path_abs = os.path.abspath(img_path)
    matching_rows = df[df['image_path'].apply(lambda x: os.path.abspath(x) == img_path_abs)]
    
    if matching_rows.empty:
        # Try to find by filename
        img_filename = os.path.basename(img_path)
        matching_rows = df[df['source_file'].str.replace('.wp', '.tiff') == img_filename]
    
    if matching_rows.empty:
        raise ValueError(f"Image {img_path} not found in dataset. Please provide a path to an image from the Nikon D810 dataset.")
    
    row = matching_rows.iloc[0]
    wp_gt = row[['mean_r', 'mean_g', 'mean_b']].values.astype(np.float32)
    cluster_name = row['cluster_name']
    
    # Process image
    raw_disp, rendered_disp, ref_name = process_image(
        row['image_path'], wp_gt, illum_wps3, illum_ccms, illum_names
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(raw_disp)
    axes[0].set_title(f"RAW (with illuminant)\n{cluster_name}\n{os.path.basename(row['image_path'])}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(rendered_disp)
    axes[1].set_title(f"Rendered (WB + CCM)\n{ref_name}\n{cluster_name}", fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(row['image_path']))[0]
    output_path = os.path.join(output_dir, f"{base_name}_preview.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def visualize_grid(num_images, df, illum_wps3, illum_ccms, illum_names, output_dir):
    """Visualize a grid of random images and save to file."""
    # Sample random images
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    sample_df = df.sample(n=min(num_images, len(df)), random_state=RANDOM_SEED)
    
    examples = []
    for _, row in sample_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
        
        try:
            wp_gt = row[['mean_r', 'mean_g', 'mean_b']].values.astype(np.float32)
            raw_disp, rendered_disp, ref_name = process_image(
                img_path, wp_gt, illum_wps3, illum_ccms, illum_names
            )
            examples.append((
                row['cluster_name'],
                ref_name,
                os.path.basename(img_path),
                raw_disp,
                rendered_disp
            ))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not examples:
        print("No examples to display.")
        return
    
    # Calculate grid dimensions
    num_examples = len(examples)
    cols = 4  # 2 pairs per row (raw + rendered)
    rows = (num_examples + cols // 2 - 1) // (cols // 2)  # Each example takes 2 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Clear all axes
    for r in range(rows):
        for c in range(cols):
            axes[r, c].axis('off')
    
    # Plot examples
    for idx, (cluster_name, ref_name, fname, raw_img, rend_img) in enumerate(examples):
        row_idx = idx // (cols // 2)
        col_idx = (idx % (cols // 2)) * 2
        
        if row_idx < rows and col_idx < cols:
            # Raw image
            axes[row_idx, col_idx].imshow(raw_img)
            axes[row_idx, col_idx].set_title(f"{cluster_name}\nRAW\n{fname}", fontsize=8)
            axes[row_idx, col_idx].axis('off')
            
            # Rendered image
            if col_idx + 1 < cols:
                axes[row_idx, col_idx + 1].imshow(rend_img)
                axes[row_idx, col_idx + 1].set_title(f"{cluster_name}\nRendered ({ref_name})\n{fname}", fontsize=8)
                axes[row_idx, col_idx + 1].axis('off')
    
    plt.tight_layout()
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"grid_{num_images}_images.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid with {len(examples)} images: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize illuminant-corrected images from Nikon D810 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview a single image
  python visualize_illuminants.py --image Data/Nikon_D810/field_1_cameras/N_D810_field_001.tiff
  
  # Show grid of 15 random images
  python visualize_illuminants.py --grid 15
  
  # Specify output directory
  python visualize_illuminants.py --grid 15 --output visualizations/
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to image file to preview')
    group.add_argument('--grid', type=int, help='Number of random images to show in grid')
    
    parser.add_argument('--output', type=str, default='visualizations', 
                       help='Output directory for saved images (default: visualizations)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_dataset()
    df, cluster_names, sorted_clusters = perform_clustering(df)
    illum_wps3, illum_ccms, illum_names = load_nikon_ccms()
    
    # Execute requested visualization
    if args.image:
        visualize_single_image(args.image, df, illum_wps3, illum_ccms, illum_names, args.output)
    elif args.grid:
        if args.grid <= 0:
            parser.error("--grid must be a positive integer")
        visualize_grid(args.grid, df, illum_wps3, illum_ccms, illum_names, args.output)

if __name__ == "__main__":
    main()
