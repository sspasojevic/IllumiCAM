"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Dataset Augmentation and Splitting Script

Processes raw illuminant data, performs K-means clustering to group images by
illuminant color, balances class distributions through augmentation, and creates
train/val/test splits. Generates the final dataset structure used for training
illuminant estimation models.

Make sure to have the Nikon_D810 dataset from INTEL-TAU in the Data/Nikon_D810 directory.
This will generate the dataset/ directory with train/val/test splits.

Usage:
python augment_split_data.py
"""

# Imports
import os
import sys
import glob
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.config import DATA_ROOT as CONFIG_DATA_ROOT, DATASET_ROOT

RANDOM_SEED = 42
SPLIT_SEED = 1337
DATA_ROOT = os.path.join(CONFIG_DATA_ROOT, "Nikon_D810")
FINAL_DATASET_DIR = DATASET_ROOT
TARGET_SIZE_STRATEGY = "max"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

def load_data(data_root):
    """
    Load .wp files and extract chromaticity.
    
    Args:
        data_root: Root directory containing .wp files
    
    Returns:
        DataFrame with chromaticity data
    """

    data_list = []
    
    search_pattern = os.path.join(data_root, "**", "*.wp")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        return pd.DataFrame()

    for wp_file in tqdm(files, desc="Reading files"):
        with open(wp_file, "r") as f:
            line = f.read().strip()
            values = line.replace("\t", " ").split()
            if len(values) >= 3:
                r, g, b = float(values[0]), float(values[1]), float(values[2])
                total = r + g + b
                if total > 0:
                    folder_name = os.path.basename(os.path.dirname(wp_file))
                    # Construct corresponding .tiff path
                    # Assuming .tiff is in the same directory with same name
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

    return pd.DataFrame(data_list)

def perform_clustering(df, n_clusters=5):
    """
    Perform KMeans clustering on chromaticity.
    
    Args:
        df: DataFrame with chromaticity data
        n_clusters: Number of clusters
    
    Returns:
        Tuple of (df_with_clusters, cluster_names, sorted_clusters)
    """

    X = df[['mean_r', 'mean_g', 'mean_b']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    sorted_clusters = sorted(range(n_clusters), key=lambda i: kmeans.cluster_centers_[i][2] / kmeans.cluster_centers_[i][0])
    
    label_template = [
        "Very_Warm",
        "Warm",
        "Neutral",
        "Cool",
        "Very_Cool"
    ]
    
    cluster_names = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        cluster_names[cluster_id] = label_template[rank] if rank < len(label_template) else f"Cluster_{rank}"

    df['cluster_name'] = df['cluster'].map(cluster_names)
        
    return df, cluster_names, sorted_clusters

def augment_image_crop_resize(img, original_size, crop_ratio_range=(0.3, 0.7)):
    """
    Random crop and resize augmentation.
    
    Args:
        img: PIL Image to augment
        original_size: Original image size (width, height)
        crop_ratio_range: Range for crop ratio
    
    Returns:
        Augmented PIL Image
    """

    width, height = original_size
    crop_ratio = np.random.uniform(*crop_ratio_range)
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    
    max_left = max(0, width - crop_w)
    max_top = max(0, height - crop_h)

    left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
    top = np.random.randint(0, max_top + 1) if max_top > 0 else 0

    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((width, height), Image.LANCZOS)
    return resized

def create_augmented_image(
    img_path,
    original_size=None,
    crop_ratio_range=(0.3, 0.7),
    flip_horizontal_prob=0.5,
    flip_vertical_prob=0.2,
):
    """
    Create one augmented version of an image.
    
    Args:
        img_path: Path to image file
        original_size: Original image size (width, height)
        crop_ratio_range: Range for crop ratio
        flip_horizontal_prob: Probability of horizontal flip
        flip_vertical_prob: Probability of vertical flip
    
    Returns:
        Augmented PIL Image
    """

    with Image.open(img_path) as img:
        if original_size is None:
            original_size = img.size

        # Crop first
        augmented = augment_image_crop_resize(img, original_size, crop_ratio_range=crop_ratio_range)

        # Flips
        if np.random.rand() < flip_horizontal_prob:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

        if np.random.rand() < flip_vertical_prob:
            augmented = augmented.transpose(Image.FLIP_TOP_BOTTOM)

        return augmented

def remove_outliers(df):
    """
    Remove outliers based on chromaticity thresholds.
    
    Args:
        df: DataFrame with chromaticity data
    
    Returns:
        Cleaned DataFrame with outliers removed
    """

    r_threshold_high = 0.5
    r_threshold_low = 0.1
    b_threshold_high = 0.6
    g_threshold_low = 0.3
    
    df_clean = df[
        (df['mean_r'] <= r_threshold_high) &
        (df['mean_r'] >= r_threshold_low) &
        (df['mean_b'] <= b_threshold_high) &
        (df['mean_g'] >= g_threshold_low)
    ].copy()
    
    return df_clean

def main():
    """
    Main function for dataset augmentation and splitting.
    """
    
    # 1. Load Data
    df = load_data(DATA_ROOT)
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # 2. Remove Outliers
    df = remove_outliers(df)
    
    # 3. Cluster
    df, cluster_names, sorted_clusters = perform_clustering(df)

    # 4. Setup Final Dataset Directories (train/val/test)
    if os.path.exists(FINAL_DATASET_DIR):
        print(f"Dataset directory {FINAL_DATASET_DIR} already exists. Please remove it first.")
        return
    
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for name in cluster_names.values():
            os.makedirs(os.path.join(FINAL_DATASET_DIR, split, name), exist_ok=True)

    # 4. Calculate Needs
    if TARGET_SIZE_STRATEGY == 'max':
        target_size = df['cluster'].value_counts().max()
    else:
        target_size = int(TARGET_SIZE_STRATEGY)

    augmentation_needs = {}
    for cluster_id in sorted_clusters:
        count = len(df[df['cluster'] == cluster_id])
        needed = max(0, target_size - count)
        augmentation_needs[cluster_id] = needed

    # 5. Process each cluster: collect images, augment, and split directly to train/val/test
    # Split ratios
    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    
    for cluster_id in sorted_clusters:
        cluster_name = cluster_names[cluster_id]
        cluster_df = df[df['cluster'] == cluster_id].copy()
        needed = augmentation_needs[cluster_id]
        
        # Collect all original images for this cluster
        all_images = []  # List of (image_path, is_original, filename)
        
        # Add original images
        for _, row in cluster_df.iterrows():
            img_path = row['image_path']
            if os.path.exists(img_path):
                source_filename = row['source_file'].replace('.wp', '.tiff')
                all_images.append((img_path, True, source_filename))
        
        # Generate augmented images and add to list
        if needed > 0:
            images_in_cluster = len(cluster_df)
            augs_per_image = needed / images_in_cluster
            aug_idx = 0
            
            for img_idx, (_, row) in enumerate(cluster_df.iterrows()):
                if aug_idx >= needed:
                    break
                    
                img_path = row['image_path']
                if not os.path.exists(img_path):
                    continue
                    
                source_filename = row['source_file'].replace('.wp', '.tiff')
                base_name = os.path.splitext(source_filename)[0]
                
                num_augs_for_this = int(np.ceil(augs_per_image * (img_idx + 1)) - np.ceil(augs_per_image * img_idx))
                num_augs_for_this = min(num_augs_for_this, needed - aug_idx)
                
                if num_augs_for_this > 0:
                    with Image.open(img_path) as original_img:
                        original_size = original_img.size
                        
                    for _ in range(num_augs_for_this):
                        aug_img = create_augmented_image(img_path, original_size)
                        aug_filename = f"{base_name}_aug_{aug_idx:04d}.tiff"
                        # Store augmented image in memory temporarily
                        all_images.append((aug_img, False, aug_filename))
                        aug_idx += 1
        
        # Shuffle images for random split
        random.seed(SPLIT_SEED)
        random.shuffle(all_images)
        
        # Split into train/val/test
        total_images = len(all_images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        # Save images to appropriate split directories
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_data, is_original, filename in tqdm(split_images, desc=f"  Saving {split_name}", leave=False):
                dest_path = os.path.join(FINAL_DATASET_DIR, split_name, cluster_name, filename)
                
                if is_original:
                    # Copy original file
                    shutil.copy2(img_data, dest_path)
                else:
                    # Save augmented image (PIL Image object)
                    img_data.save(dest_path)

if __name__ == "__main__":
    main()

