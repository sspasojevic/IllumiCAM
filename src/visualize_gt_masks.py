"""
Visualize ground truth masks for LSMI_Test_Package images.
Takes an LSMI NEF image path and displays the ground truth illuminant masks.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.lsmi_utils import process_raw_image, load_mask, CLUSTER_NAMES
from src.data_loader import get_datasets

IMG_SIZE = 224
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")


def visualize_gt_masks(image_path):
    """Visualize ground truth masks for an LSMI image."""
    # Check if this is from LSMI_Test_Package
    lsmi_masks_dir = os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package", "masks")
    
    # Load the image
    print(f"Loading image: {image_path}")
    rgb_array = process_raw_image(image_path, srgb=False)
    
    # Resize image to IMG_SIZE for display
    rgb_image = cv2.resize(rgb_array, (IMG_SIZE, IMG_SIZE))
    
    # Load ground truth mask
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(lsmi_masks_dir, f"{img_basename}_mask.npy")
    
    print(f"Loading ground truth mask: {mask_path}")
    gt_masks = load_mask(mask_path, target_shape=(IMG_SIZE, IMG_SIZE))
    print(f"Loaded mask shape: {gt_masks.shape}")
    
    # Get label names for display
    _, _, _, label_names = get_datasets()
    
    # Create visualization
    # Mask channel order: ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
    MASK_CHANNEL_ORDER = CLUSTER_NAMES
    
    cols = 1 + len(label_names)  # Original + GT masks for each class
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    
    col_idx = 0
    
    # Original image
    axes[col_idx].imshow(rgb_image)
    axes[col_idx].set_title("Original Image", fontsize=12, weight='bold')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # Ground truth masks for each class
    for class_name in label_names:
        ax = axes[col_idx]
        # Find corresponding mask channel index
        mask_ch_idx = MASK_CHANNEL_ORDER.index(class_name)
        mask_channel = gt_masks[:, :, mask_ch_idx]
        # Normalize mask to 0-1 for display
        if mask_channel.max() > 0:
            mask_channel_norm = mask_channel / mask_channel.max()
        else:
            mask_channel_norm = mask_channel
        
        # Apply JET colormap to mask (0-255 range)
        mask_colored = cv2.applyColorMap(
            np.uint8(255 * mask_channel_norm), 
            cv2.COLORMAP_JET
        )
        # Convert BGR to RGB for matplotlib
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
        # Overlay on original image (50/50 blend)
        mask_overlay = (rgb_image * 0.5 + mask_colored * 0.5).astype(np.uint8)
        ax.imshow(mask_overlay)
        
        ax.set_title(f"GT Mask\n{class_name}", fontsize=12, color='green', weight='bold')
        ax.axis('off')
        col_idx += 1
    
    plt.tight_layout()
    
    # Save visualization
    gt_dir = os.path.join(VISUALIZATIONS_DIR, "gt_masks")
    os.makedirs(gt_dir, exist_ok=True)
    save_path = os.path.join(gt_dir, f"{img_basename}_gt_masks.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved ground truth mask visualization to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ground truth masks for LSMI images')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to LSMI NEF image')
    
    args = parser.parse_args()
    visualize_gt_masks(args.image)


if __name__ == "__main__":
    main()

