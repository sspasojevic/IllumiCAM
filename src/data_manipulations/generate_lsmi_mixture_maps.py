"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

LSMI Mixture Maps Generation Script

Generates ground truth mixture maps for the LSMI dataset.
Used by prepare_lsmi_test_package.py to generate the test package.

Usage:
python generate_lsmi_mixture_maps.py
"""

# Imports
import os
import json
import cv2
import rawpy
import numpy as np
from tqdm import tqdm
import argparse

# Constants
BLACK_LEVEL = 0 # Will be read from raw
SATURATION = 0  # Will be read from raw
ZERO_MASK = -1

def get_coefficient_map(img_1_wb, img_2_wb, zero_mask=-1):
    """
    Calculate illuminant coefficient map from two white-balanced images.
    
    Args:
        img_1_wb: First white-balanced image
        img_2_wb: Second white-balanced image
        zero_mask: Masking value for pixels where both G = 0
    
    Returns:
        Coefficient map for img_1's illuminant
    """

    denominator = img_1_wb[:,:,1] + img_2_wb[:,:,1]
    coefficient = img_1_wb[:,:,1] / np.clip(denominator, 0.0001, None)
    coefficient = np.where(denominator==0, zero_mask, coefficient)
    return coefficient

def process_place(place_path, place_name, place_info):
    """
    Process a single place to generate mixture map.
    
    Args:
        place_path: Path to place directory
        place_name: Name of the place
        place_info: Dictionary containing place metadata
    """

    num_lights = place_info["NumOfLights"]
    
    # Paths
    img_1_path = os.path.join(place_path, f"{place_name}_1.nef")
    
    if num_lights == 2:
        img_12_path = os.path.join(place_path, f"{place_name}_12.nef")
        
        if not os.path.exists(img_1_path) or not os.path.exists(img_12_path):
            return

        # Read RAWs
        with rawpy.imread(img_1_path) as raw_1:
            img_1 = raw_1.raw_image.copy().astype("int16")
            black_level = min(raw_1.black_level_per_channel)
            saturation = raw_1.white_level
            
        with rawpy.imread(img_12_path) as raw_12:
            img_12 = raw_12.raw_image.copy().astype("int16")

        # Subtract black level
        img_1 = np.clip(img_1 - black_level, 0, saturation - black_level)
        img_12 = np.clip(img_12 - black_level, 0, saturation - black_level)
        
        # Calculate img_2
        img_2 = np.clip(img_12 - img_1, 0, saturation - black_level)

        # Demosaic raw images to linear RGB for coefficient calculation
        with rawpy.imread(img_1_path) as raw:
            rgb_1 = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, user_wb=[1,1,1,1], user_black=0)
            
        with rawpy.imread(img_12_path) as raw:
            rgb_12 = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, user_wb=[1,1,1,1], user_black=0)
        
        # Calculate Light 2
        rgb_2 = np.clip(rgb_12.astype(np.int32) - rgb_1.astype(np.int32), 0, 65535).astype(np.uint16)
        
        # Calculate coefficients using Green channel
        
        denom = rgb_12[:,:,1].astype(np.float32)
        g1 = rgb_1[:,:,1].astype(np.float32)
        
        # Avoid division by zero
        mask = denom > 0
        coeff_1 = np.zeros_like(denom)
        coeff_1[mask] = g1[mask] / denom[mask]
        
        # Clip to [0, 1]
        coeff_1 = np.clip(coeff_1, 0, 1)
        
        coeff_2 = 1.0 - coeff_1
        coeff_2[~mask] = 0
        
        # Save as (H, W, 2)
        mixture_map = np.stack([coeff_1, coeff_2], axis=-1)
        
        output_path = os.path.join(place_path, f"{place_name}_mixture.npy")
        np.save(output_path, mixture_map)
        
    elif num_lights == 3:
        img_12_path = os.path.join(place_path, f"{place_name}_12.nef")
        img_13_path = os.path.join(place_path, f"{place_name}_13.nef")
        img_123_path = os.path.join(place_path, f"{place_name}_123.nef")
        
        if not all(os.path.exists(p) for p in [img_1_path, img_12_path, img_13_path, img_123_path]):
             return

        # Load and process to linear RGB
        def load_linear(path):
            with rawpy.imread(path) as raw:
                return raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, user_wb=[1,1,1,1], user_black=0)

        rgb_1 = load_linear(img_1_path).astype(np.int32)
        rgb_12 = load_linear(img_12_path).astype(np.int32)
        rgb_13 = load_linear(img_13_path).astype(np.int32)
        rgb_123 = load_linear(img_123_path).astype(np.int32)
        
        # Derive individual lights
        rgb_2 = np.clip(rgb_12 - rgb_1, 0, 65535)
        rgb_3 = np.clip(rgb_13 - rgb_1, 0, 65535)
        
        # Calculate coefficients using sum of components for consistency
        denom = rgb_1[:,:,1] + rgb_2[:,:,1] + rgb_3[:,:,1]
        denom = denom.astype(np.float32)
        
        mask = denom > 0
        
        c1 = np.zeros_like(denom)
        c2 = np.zeros_like(denom)
        c3 = np.zeros_like(denom)
        
        c1[mask] = rgb_1[:,:,1][mask] / denom[mask]
        c2[mask] = rgb_2[:,:,1][mask] / denom[mask]
        c3[mask] = rgb_3[:,:,1][mask] / denom[mask]
        
        # Clip to [0, 1]
        c1 = np.clip(c1, 0, 1)
        c2 = np.clip(c2, 0, 1)
        c3 = np.clip(c3, 0, 1)
        
        mixture_map = np.stack([c1, c2, c3], axis=-1)
        output_path = os.path.join(place_path, f"{place_name}_mixture.npy")
        np.save(output_path, mixture_map)

def main():
    """
    Main function to generate LSMI mixture maps.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Data/LSMI/nikon")
    args = parser.parse_args()
    
    meta_path = os.path.join(args.root, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    places = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
    
    for place in tqdm(places):
        if place not in meta:
            continue
            
        process_place(os.path.join(args.root, place), place, meta[place])

if __name__ == "__main__":
    main()
