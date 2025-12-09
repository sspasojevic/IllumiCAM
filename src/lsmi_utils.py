import rawpy
import numpy as np
import cv2
import os

def process_raw_image(raw_path, srgb=False):
    """
    Reads a RAW image (NEF) and converts it to RGB.
    
    Args:
        raw_path (str): Path to the .nef file.
        srgb (bool): If True, returns sRGB image (gamma corrected, white balanced).
                     If False, returns linear RGB (camera color space, no auto brightness).
                     
    Returns:
        np.ndarray: RGB image. 
                    If srgb=True, dtype is usually uint8 (0-255) or float (0-1) depending on rawpy version/settings.
                    If srgb=False, dtype is usually uint16 or float.
    """
    with rawpy.imread(raw_path) as raw:
        # Use half_size=True for speed, consistent with dataset generation
        if srgb:
             # Standard sRGB output: Camera WB, Auto Brightness (or not, usually better to disable for consistency but for vis we want it nice)
             # Note: The dataset generation used use_camera_wb=True, no_auto_bright=False for sRGB visualization
             rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=False)
        else:
             # Linear/Raw output: No WB (or unitary), No Auto Brightness, Raw Color Space
             # Note: The dataset generation used use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw
             rgb = raw.postprocess(half_size=True, use_camera_wb=False, user_wb=[1,1,1,1], no_auto_bright=True, output_color=rawpy.ColorSpace.raw)
    return rgb

def load_mask(mask_path, target_shape=None):
    """
    Loads a ground truth mask and optionally resizes it to match a target image shape.
    
    Args:
        mask_path (str): Path to the .npy mask file.
        target_shape (tuple): (Height, Width) of the target image.
        
    Returns:
        np.ndarray: Mask of shape (H, W, 5). Channels correspond to clusters:
                    ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
        
    mask = np.load(mask_path)
    
    if target_shape is not None:
        h, w = target_shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
    return mask

CLUSTER_NAMES = ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']

def get_cluster_names():
    return CLUSTER_NAMES

