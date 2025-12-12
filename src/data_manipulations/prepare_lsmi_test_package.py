"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

LSMI Test Package Preparation Script

Prepares the LSMI test package by generating mixture maps and copying necessary files.
Uses the generate_lsmi_mixture_maps.py script to generate the mixture maps. It also uses
the cluster centers from the cluster_centers.npy file.

Usage:
python prepare_lsmi_test_package.py --csv lsmi_balanced.csv --src_root Data/LSMI/nikon --output_dir LSMI_Test_Package --meta Data/LSMI/nikon/meta.json
"""

# Imports
import os
import shutil
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import rawpy
import sys

# Import generation logic
try:
    import generate_lsmi_mixture_maps
except ImportError:
    sys.path.append(os.getcwd())
    import generate_lsmi_mixture_maps

# Add repo root to path to import config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

from config.config import LSMI_DATASET_ROOT

def main():
    """
    Main function to prepare LSMI test package.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lsmi_balanced.csv", help="Path to balanced CSV")
    parser.add_argument("--src_root", default=os.path.join(LSMI_DATASET_ROOT, "nikon"), help="Source dataset path")
    parser.add_argument("--output_dir", default="LSMI_Test_Package", help="Output package path")
    parser.add_argument("--meta", default=os.path.join(LSMI_DATASET_ROOT, "nikon", "meta.json"), help="Path to original meta.json")
    args = parser.parse_args()

    # Create output directories
    images_dir = os.path.join(args.output_dir, "images")
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Read CSV
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    target_places = df['place'].unique()
    print(f"Found {len(target_places)} unique places in CSV.")

    # Read Meta
    print(f"Reading {args.meta}...")
    with open(args.meta, 'r') as f:
        full_meta = json.load(f)

    subset_meta = {}
    
    # Process each place
    print("Processing places...")
    for place in tqdm(target_places):
        if place not in full_meta:
            print(f"Warning: {place} not found in meta.json. Skipping.")
            continue

        place_info = full_meta[place]
        num_lights = place_info["NumOfLights"]
        
        # Filter for 2 or 3 illuminants
        if num_lights not in [2, 3]:
            print(f"Skipping {place}: NumOfLights is {num_lights}")
            continue

        # Add to subset meta
        subset_meta[place] = place_info

        # Source directory
        src_place_dir = os.path.join(args.src_root, place)
        if not os.path.exists(src_place_dir):
            print(f"Warning: Directory {src_place_dir} not found.")
            continue

        # Copy mask file
        src_mask = os.path.join(src_place_dir, f"{place}_mixture.npy")
        dst_mask = os.path.join(masks_dir, f"{place}.npy")
        
        # Generate mask if missing
        if not os.path.exists(src_mask):
             print(f"Mask {src_mask} missing. Generating...")
             generate_lsmi_mixture_maps.process_place(src_place_dir, place, place_info)
        
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"Error: Failed to generate mask for {place}")

        # Copy image file
        src_image = None
        if num_lights == 2:
            src_image = os.path.join(src_place_dir, f"{place}_12.nef")
        elif num_lights == 3:
            src_image = os.path.join(src_place_dir, f"{place}_123.nef")
            
        if src_image and os.path.exists(src_image):
            dst_image = os.path.join(images_dir, f"{place}.nef")
            shutil.copy2(src_image, dst_image)
        else:
            print(f"Warning: Image {src_image} missing.")

    # Save subset meta
    print("Saving subset meta.json...")
    with open(os.path.join(args.output_dir, "meta.json"), 'w') as f:
        json.dump(subset_meta, f, indent=4)

    # Copy cluster centers
    cluster_src = "cluster_centers.npy"
    if os.path.exists(cluster_src):
        shutil.copy2(cluster_src, os.path.join(args.output_dir, "cluster_centers.npy"))
    else:
        print("Warning: cluster_centers.npy not found in current directory.")

if __name__ == "__main__":
    main()
