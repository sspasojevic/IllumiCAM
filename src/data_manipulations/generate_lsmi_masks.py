"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

LSMI Ground Truth Mask Generation Script

Generates pixel-level ground truth masks for LSMI (Localized Spatially Mixed Illuminant)
test images. Processes RAW NEF files with ColorChecker charts, extracts illuminant
information from chart cells, and creates multi-channel masks indicating which pixels
are illuminated by which cluster. Each mask has 5 channels corresponding to illuminant
clusters: Very_Warm, Warm, Neutral, Cool, Very_Cool.

Uses:
    - config.config for data paths and cluster centers
    - rawpy for RAW image processing
    - OpenCV for perspective transforms and mask generation
"""

import os
import sys
import json
import numpy as np
import rawpy
import cv2
import argparse
from tqdm import tqdm
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.config import DATA_ROOT, CLUSTER_CENTERS_PATH

# Constants from 1_make_mixture_map.py
CELLCHART = np.float32([
    # Row 1
    [0.25, 0.25],   [2.75, 0.25],   [2.75, 2.75],   [0.25, 2.75],
    [3.00, 0.25],   [5.50, 0.25],   [5.50, 2.75],   [3.00, 2.75], 
    [5.75, 0.25],   [8.25, 0.25],   [8.25, 2.75],   [5.75, 2.75],
    [8.50, 0.25],   [11.00, 0.25],  [11.00, 2.75],  [8.50, 2.75],
    [11.25, 0.25],  [13.75, 0.25],  [13.75, 2.75],  [11.25, 2.75],
    [14.00, 0.25],  [16.50, 0.25],  [16.50, 2.75],  [14.00, 2.75],
    # Row 2  
    [0.25, 3.00],   [2.75, 3.00],   [2.75, 5.50],   [0.25, 5.50],
    [3.00, 3.00],   [5.50, 3.00],   [5.50, 5.50],   [3.00, 5.50],
    [5.75, 3.00],   [8.25, 3.00],   [8.25, 5.50],   [5.75, 5.50],
    [8.50, 3.00],   [11.00, 3.00],  [11.00, 5.50],  [8.50, 5.50],
    [11.25, 3.00],  [13.75, 3.00],  [13.75, 5.50],  [11.25, 5.50],
    [14.00, 3.00],  [16.50, 3.00],  [16.50, 5.50],  [14.00, 5.50],
    # Row 3
    [0.25, 5.75],   [2.75, 5.75],   [2.75, 8.25],   [0.25, 8.25],
    [3.00, 5.75],   [5.50, 5.75],   [5.50, 8.25],   [3.00, 8.25],
    [5.75, 5.75],   [8.25, 5.75],   [8.25, 8.25],   [5.75, 8.25],
    [8.50, 5.75],   [11.00, 5.75],  [11.00, 8.25],  [8.50, 8.25],
    [11.25, 5.75],  [13.75, 5.75],  [13.75, 8.25],  [11.25, 8.25],
    [14.00, 5.75],  [16.50, 5.75],  [16.50, 8.25],  [14.00, 8.25],
    # Row 4
    [0.25, 8.50],   [2.75, 8.50],   [2.75, 11.00],  [0.25, 11.00],
    [3.00, 8.50],   [5.50, 8.50],   [5.50, 11.00],  [3.00, 11.00],
    [5.75, 8.50],   [8.25, 8.50],   [8.25, 11.00],  [5.75, 11.00],
    [8.50, 8.50],   [11.00, 8.50],  [11.00, 11.00], [8.50, 11.00],
    [11.25, 8.50],  [13.75, 8.50],  [13.75, 11.00], [11.25, 11.00],
    [14.00, 8.50],  [16.50, 8.50],  [16.50, 11.00], [14.00, 11.00]
])
MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])

def manual_perspective_transform(points, h):
    points = np.array(points)
    # print(f"DEBUG: points.shape={points.shape}")
    if len(points.shape) != 2:
        # print(f"DEBUG: Reshaping points from {points.shape}")
        points = points.reshape(-1, 2)
    
    # Add z=1
    try:
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    except Exception as e:
        print(f"Error in hstack: {e}. points.shape={points.shape}")
        raise e
        
    # Transform
    transformed = points_homo @ h.T
    # Normalize
    transformed /= transformed[:, 2:3]
    return transformed[:, :2]

def angular_error(x, y):
    x = np.array(x)
    y = np.array(y)
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    x_norm = np.maximum(x_norm, 1e-8)
    y_norm = np.maximum(y_norm, 1e-8)
    x = x / x_norm
    y = y / y_norm
    dot = np.sum(x * y, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def map_illuminant_to_cluster(illuminant_rgb, cluster_centers):
    errors = [angular_error(illuminant_rgb, center) for center in cluster_centers]
    return np.argmin(errors)

def load_cluster_centers(path):
    data = np.load(path, allow_pickle=True)
    if data.shape == ():
        data = data.item()
    keys = ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
    centers = []
    for k in keys:
        if k in data:
            centers.append(data[k])
    return np.array(centers)

def get_patch_chroma(img, mcc_coord):
    """
    Extract illuminant chromaticity from image using MCC coordinates.
    """
    # Calculate homography
    # Check if coords need scaling
    h, w = img.shape[:2]
    # If coords are outside image bounds, assume they are for full resolution and scale down
    if np.any(mcc_coord > np.array([w, h])):
        mcc_coord = mcc_coord * 0.5
        
    h_matrix = cv2.getPerspectiveTransform(MCCBOX, mcc_coord)
    if h_matrix is None:
        print(f"Error: h_matrix is None for mcc_coord: {mcc_coord}")
        return np.array([0,0,0])
    
    gray_patches_indices = [18, 19, 20, 21, 22, 23]
    
    patch_colors = []
    
    for idx in gray_patches_indices:
        # Get 4 corners of the patch from CELLCHART
        corners_src = CELLCHART[idx*4 : (idx+1)*4]
        # Transform to image coordinates
        try:
            corners_dst = manual_perspective_transform(corners_src, h_matrix)
        except Exception as e:
            print(f"Error in manual_perspective_transform: {e}")
            continue
        
        # Create mask for the patch
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners_dst.astype(np.int32), 1)
        
        # Extract mean color
        mean_val = cv2.mean(img, mask=mask)[:3] # BGR or RGB depending on input
        patch_colors.append(mean_val)
        
    if not patch_colors:
        print("Warning: No patches extracted.")
        return np.array([0,0,0])

    # Average the gray patches (excluding saturation if needed, but we assume linear raw is safe-ish)
    # Or use the brightest non-saturated one.
    # Let's just average all of them for robustness.
    avg_color = np.mean(patch_colors, axis=0)
    
    # Normalize (G=1 or Sum=1)
    # We use Sum=1 (chromaticity)
    s = np.sum(avg_color)
    if s > 0:
        return avg_color / s
    return avg_color

def process_place(place_path, place_name, meta, cluster_centers, output_dir):
    num_lights = meta["NumOfLights"]
    mcc_coords = meta["MCCCoord"]
    
    # Load images
    # Assuming Nikon naming convention: PlaceX_1.nef, PlaceX_12.nef, etc.
    try:
        raw_1 = rawpy.imread(os.path.join(place_path, f"{place_name}_1.nef"))
        img_1 = raw_1.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
        
        if num_lights == 2:
            raw_12 = rawpy.imread(os.path.join(place_path, f"{place_name}_12.nef"))
            img_12 = raw_12.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
            
            # Subtract to get Light 2
            img_2 = np.clip(img_12.astype(np.int32) - img_1.astype(np.int32), 0, 65535).astype(np.uint16)
            
            # Calculate Chromaticities
            # MCC coords are float32 array
            mcc1 = np.float32(mcc_coords["mcc1"])
            mcc2 = np.float32(mcc_coords["mcc2"])
            
            chroma_1 = get_patch_chroma(img_1, mcc1)
            chroma_2 = get_patch_chroma(img_2, mcc2)
            
            # Map to clusters
            c1_idx = map_illuminant_to_cluster(chroma_1, cluster_centers)
            c2_idx = map_illuminant_to_cluster(chroma_2, cluster_centers)
            
            # Calculate Mixture Weights (Green Channel ONLY - channel index 1)
            # Following LSMI methodology: weights = green_channel_intensity / sum(green_channels)
            green_1 = img_1[:,:,1].astype(float)
            green_2 = img_2[:,:,1].astype(float)
            denom = green_1 + green_2
            
            # Safety check for zero denominator (ZERO_MASK concept from LSMI)
            # Dark pixels or shadows will have denominator near 0, which causes NaN/Inf
            # We set these pixels to 0 weight (no contribution to any cluster)
            epsilon = 1e-6
            valid_mask = denom > epsilon
            
            # Initialize weights to 0 (ZERO_MASK for invalid pixels)
            w1 = np.zeros_like(denom, dtype=np.float32)
            w2 = np.zeros_like(denom, dtype=np.float32)
            
            # Calculate weights only for valid pixels
            w1[valid_mask] = green_1[valid_mask] / denom[valid_mask]
            w2[valid_mask] = green_2[valid_mask] / denom[valid_mask]
            
            # Enforce [0, 1] range using np.clip for additional safety
            w1 = np.clip(w1, 0.0, 1.0)
            w2 = np.clip(w2, 0.0, 1.0)
            
            # Create Mask (5 channels)
            mask = np.zeros((img_1.shape[0], img_1.shape[1], 5), dtype=np.float32)
            mask[:,:,c1_idx] += w1
            mask[:,:,c2_idx] += w2
            
            # Save
            np.save(os.path.join(output_dir, f"{place_name}_mask.npy"), mask)
            
        elif num_lights == 3:
            raw_12 = rawpy.imread(os.path.join(place_path, f"{place_name}_12.nef"))
            img_12 = raw_12.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
            
            raw_13 = rawpy.imread(os.path.join(place_path, f"{place_name}_13.nef"))
            img_13 = raw_13.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
            
            img_2 = np.clip(img_12.astype(np.int32) - img_1.astype(np.int32), 0, 65535).astype(np.uint16)
            img_3 = np.clip(img_13.astype(np.int32) - img_1.astype(np.int32), 0, 65535).astype(np.uint16)
            
            mcc1 = np.float32(mcc_coords["mcc1"])
            mcc2 = np.float32(mcc_coords["mcc2"])
            mcc3 = np.float32(mcc_coords["mcc3"])
            
            chroma_1 = get_patch_chroma(img_1, mcc1)
            chroma_2 = get_patch_chroma(img_2, mcc2)
            chroma_3 = get_patch_chroma(img_3, mcc3)
            
            c1_idx = map_illuminant_to_cluster(chroma_1, cluster_centers)
            c2_idx = map_illuminant_to_cluster(chroma_2, cluster_centers)
            c3_idx = map_illuminant_to_cluster(chroma_3, cluster_centers)
            
            # Calculate Mixture Weights (Green Channel ONLY - channel index 1)
            # Following LSMI methodology: weights = green_channel_intensity / sum(green_channels)
            green_1 = img_1[:,:,1].astype(float)
            green_2 = img_2[:,:,1].astype(float)
            green_3 = img_3[:,:,1].astype(float)
            denom = green_1 + green_2 + green_3
            
            # Safety check for zero denominator (ZERO_MASK concept from LSMI)
            # Dark pixels or shadows will have denominator near 0, which causes NaN/Inf
            # We set these pixels to 0 weight (no contribution to any cluster)
            epsilon = 1e-6
            valid_mask = denom > epsilon
            
            # Initialize weights to 0 (ZERO_MASK for invalid pixels)
            w1 = np.zeros_like(denom, dtype=np.float32)
            w2 = np.zeros_like(denom, dtype=np.float32)
            w3 = np.zeros_like(denom, dtype=np.float32)
            
            # Calculate weights only for valid pixels
            w1[valid_mask] = green_1[valid_mask] / denom[valid_mask]
            w2[valid_mask] = green_2[valid_mask] / denom[valid_mask]
            w3[valid_mask] = green_3[valid_mask] / denom[valid_mask]
            
            # Enforce [0, 1] range using np.clip for additional safety
            w1 = np.clip(w1, 0.0, 1.0)
            w2 = np.clip(w2, 0.0, 1.0)
            w3 = np.clip(w3, 0.0, 1.0)
            
            mask = np.zeros((img_1.shape[0], img_1.shape[1], 5), dtype=np.float32)
            mask[:,:,c1_idx] += w1
            mask[:,:,c2_idx] += w2
            mask[:,:,c3_idx] += w3
            
            np.save(os.path.join(output_dir, f"{place_name}_mask.npy"), mask)
            
    except Exception as e:
        print(f"Error processing {place_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsmi_root", type=str, default=os.path.join(DATA_ROOT, "LSMI", "nikon"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(DATA_ROOT, "LSMI", "masks"))
    parser.add_argument("--centroids_file", type=str, default=CLUSTER_CENTERS_PATH)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cluster_centers = load_cluster_centers(args.centroids_file)
    print(f"Loaded cluster centers. Shape: {cluster_centers.shape}")
    
    with open(os.path.join(args.lsmi_root, "meta.json"), "r") as f:
        meta_data = json.load(f)
        
    places = sorted([d for d in os.listdir(args.lsmi_root) if os.path.isdir(os.path.join(args.lsmi_root, d))])
    
    print(f"Found {len(places)} places.")
    
    for place in tqdm(places):
        if place in meta_data:
            process_place(os.path.join(args.lsmi_root, place), place, meta_data[place], cluster_centers, args.output_dir)
        else:
            print(f"Skipping {place} (no meta data)")

if __name__ == "__main__":
    main()
