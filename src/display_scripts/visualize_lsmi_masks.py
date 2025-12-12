import os
import sys
import numpy as np
import matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import argparse
import json
import rawpy

# Add repo root to path to import config
# Assuming we are in IllumiCAM/src/display_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../src/display_scripts
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # .../IllumiCAM
sys.path.insert(0, REPO_ROOT)

from config.config import LSMI_TEST_PACKAGE, LSMI_MASKS_DIR, CLUSTER_CENTERS_PATH, VISUALIZATIONS_DIR

def load_cluster_centers(path):
    data = np.load(path, allow_pickle=True)
    if data.shape == ():
        data = data.item()
    return data

def get_nearest_cluster(rgb, centers):
    min_dist = float('inf')
    best_name = "Unknown"
    query = np.array(rgb)
    for name, center in centers.items():
        c = np.array(center)
        dist = np.linalg.norm(query - c)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

def process_raw_image(raw_path):
    # Using similar logic to original script: linear RGB for visualization?
    # Original script used: raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, user_wb=[1,1,1,1])
    # visualize_gt_masks.py uses: raw.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
    # I will stick to the original script logic as requested ("keep the logic same as the current one")
    # But I will use half_size=True if the mask is half size? 
    # The generated masks were full size I believe.
    # Let's use full size to be safe, or check dimensions.
    
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8, user_wb=[1,1,1,1])
    return rgb

def visualize_lsmi_masks(image_path, package_path=None):
    # Determine paths
    if package_path:
        test_package = package_path
        masks_dir = os.path.join(test_package, "masks")
    else:
        test_package = LSMI_TEST_PACKAGE
        masks_dir = LSMI_MASKS_DIR

    # Derive paths
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    place_name = img_basename # Assuming filename is PlaceX.nef
    
    # Load meta
    meta_path = os.path.join(test_package, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    if place_name not in meta:
        print(f"Error: {place_name} not found in meta.json")
        return

    place_info = meta[place_name]
    num_lights = place_info["NumOfLights"]
    
    # Load Image
    print(f"Loading image: {image_path}")
    img = process_raw_image(image_path)
    
    # Load Mask
    mask_path = os.path.join(masks_dir, f"{place_name}.npy")
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found at {mask_path}")
        return
        
    print(f"Loading mask: {mask_path}")
    mask = np.load(mask_path)
    
    # Resize mask if needed
    if mask.shape[:2] != img.shape[:2]:
        print(f"Resizing mask from {mask.shape} to {img.shape}")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Re-normalize if resizing introduced artifacts? 
        # Linear interpolation is fine for visualization.

    # Load Centers
    # Try loading from package first
    centers_path = os.path.join(test_package, "cluster_centers.npy")
    if not os.path.exists(centers_path):
        centers_path = CLUSTER_CENTERS_PATH
        
    if os.path.exists(centers_path):
        centers = load_cluster_centers(centers_path)
    else:
        print("Warning: cluster_centers.npy not found.")
        centers = {}

    # Create figure
    fig, axes = plt.subplots(num_lights, 1, figsize=(10, 5 * num_lights))
    if num_lights == 1:
        axes = [axes]
        
    for i in range(num_lights):
        if i >= mask.shape[2]:
            break
            
        ax = axes[i]
        
        # Get mask channel
        m = mask[:,:,i]
        
        # Overlay
        ax.imshow(img)
        ax.imshow(m, cmap='jet', alpha=0.5)
        
        # Get illuminant info
        light_key = f"Light{i+1}"
        light_chroma = place_info[light_key] # [r, g, b]
        
        # Find cluster
        cluster_name = get_nearest_cluster(light_chroma, centers)
        
        ax.set_title(f"{place_name} - Light {i+1} - Cluster: {cluster_name}")
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(VISUALIZATIONS_DIR, "lsmi_masks")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{place_name}_vis.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to NEF image")
    parser.add_argument("--package_path", help="Path to LSMI Test Package root (overrides config)")
    args = parser.parse_args()
    
    visualize_lsmi_masks(args.image, args.package_path)

if __name__ == "__main__":
    main()
