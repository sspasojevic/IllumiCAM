"""
Unified CAM Visualization Tool
Supports: GradCAM, GradCAM++, ScoreCAM
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import all CAM methods
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import project modules
# Hardcoded configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 5
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
IMG_SIZE = 224
from src.models.model import IlluminantCNN
from src.models.model_confidence import ConfidenceWeightedCNN
from src.models.model_paper import ColorConstancyCNN
from src.models.model_illumicam3 import IllumiCam3
from src.data_loader import get_datasets, get_dataloaders


# Model registry
MODELS = {
    'standard': IlluminantCNN,
    'confidence': ConfidenceWeightedCNN,
    'paper': lambda: ColorConstancyCNN(K=NUM_CLASSES, pretrained=False),
    'illumicam3': IllumiCam3
}

# Hardcoded model paths
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_PATHS = {
    'standard': os.path.join(SAVED_MODELS_DIR, "best_illuminant_cnn_val_8084.pth"),
    'confidence': os.path.join(SAVED_MODELS_DIR, "best_illuminant_cnn_confidence.pth"),
    'paper': os.path.join(SAVED_MODELS_DIR, "best_paper_model.pth"),
    'illumicam3': os.path.join(SAVED_MODELS_DIR, "best_illumicam3.pth")
}


# CAM registry
CAM_METHODS = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'scorecam': ScoreCAM
}


class ModelWrapper(torch.nn.Module):
    """Wrapper to make models compatible with pytorch_grad_cam."""
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.model_type = model_type
    
    def forward(self, x):
        if self.model_type == 'confidence':
            logits, _ = self.model(x)
            return logits
        else:
            # standard and paper models return logits directly
            return self.model(x)


def get_available_layers(model, model_type):
    """Extract available convolutional layers from model."""
    layers = []
    
    if model_type == 'standard':
        layers = [
            ('conv1', model.conv1),
            ('conv2', model.conv2),
            ('conv3', model.conv3),
            ('conv4', model.conv4),
            ('conv5', model.conv5),
        ]
    elif model_type == 'confidence':
        layers = [
            ('conv1', model.conv1),
            ('conv2', model.conv2),
            ('conv3', model.conv3),
            ('conv4', model.conv4),
            ('conv5', model.conv5),
        ]
    elif model_type == 'illumicam3':
        layers = [
            ('conv1', model.conv1),
            ('conv2', model.conv2),
            ('conv3', model.conv3),
            ('conv4', model.conv4),
            ('conv5', model.conv5),
        ]
    elif model_type == 'paper':
        # AlexNet features: conv layers at indices 0, 3, 6, 8, 10
        layers = [
            ('conv1', model.features[0]),  # Conv2d(3, 64, ...)
            ('conv2', model.features[3]),  # Conv2d(64, 192, ...)
            ('conv3', model.features[6]),  # Conv2d(192, 384, ...)
            ('conv4', model.features[8]),  # Conv2d(384, 256, ...)
            ('conv5', model.features[10]), # Conv2d(256, 256, ...)
        ]
    
    return layers


def tensor_to_rgb(img_tensor, mean, std):
    """Convert normalized tensor to RGB [0,1] for visualization."""
    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np.astype(np.float32)


def load_image_from_path(img_path):
    """Load and preprocess a single image from path."""
    from torchvision import transforms
    import numpy as np
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    # Handle NEF files using rawpy if available
    if img_path.lower().endswith('.nef'):
        try:
            # Add project root to path to import lsmi_utils
            lsmi_utils_path = os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package", "lsmi_utils.py")
            if os.path.exists(lsmi_utils_path):
                import sys
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package"))
                try:
                    import rawpy
                    # Load ACTUAL raw sensor data (Bayer pattern) - no processing, no demosaicing
                    # This is the true raw sensor output before any processing
                    with rawpy.imread(img_path) as raw:
                        # Get the raw Bayer pattern data directly from the sensor
                        # raw.raw_image is the actual raw sensor data (Bayer pattern)
                        raw_bayer = raw.raw_image.copy()  # This is the actual raw sensor data
                        
                        # raw_bayer is a 2D array with the Bayer pattern (RGGB)
                        # We need to convert this to RGB for display, but preserve the raw sensor characteristics
                        # Option 1: Simple demosaicing to visualize (but keep raw characteristics)
                        # Option 2: Use postprocess with minimal processing to get RGB from Bayer
                        
                        # Use postprocess with raw color space, no WB, no auto brightness
                        # This gives us RGB from the Bayer pattern but preserves raw sensor characteristics
                        rgb_array = raw.postprocess(
                            half_size=True,
                            use_camera_wb=False,
                            user_wb=[1, 1, 1, 1],  # No white balance
                            no_auto_bright=True,   # No auto brightness
                            output_color=rawpy.ColorSpace.raw,  # Raw color space
                            output_bps=8  # 8-bit output
                        )
                    
                    # Convert to the exact format matching training data
                    # Training images are uint8 with very low values (dark green appearance)
                    # This is raw sensor data that hasn't been white-balanced or tone-mapped
                    # We use the robust scaling logic from apply_ccm_with_cam.py
                    if True: # simplify block structure
                        # Always convert to float for precise scaling
                        rgb_array = rgb_array.astype(np.float32)
                        
                        # Match the mean brightness of training images (~17 out of 255)
                        current_mean = rgb_array.mean()
                        target_mean = 17.0  # Average mean of training images
                        
                        if current_mean > 0:
                            # Scale so mean matches training data exactly
                            scale_factor = target_mean / (current_mean + 1e-12)
                            rgb_array = rgb_array * scale_factor
                        
                        # Clip to uint8 range (0-255)
                        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
                    
                    # This is now the actual raw sensor data converted to RGB (from Bayer pattern)
                    # It preserves the raw sensor characteristics: dark green, no WB, linear RGB
                    img = Image.fromarray(rgb_array)
                except ImportError:
                    # rawpy not available, use PIL but make it dark to match training data
                    print("Warning: rawpy not available, using PIL for NEF (may not match training format exactly)")
                    img = Image.open(img_path).convert('RGB')
                    # PIL opens NEF as bright RGB, scale down to match training data darkness
                    img_array = np.array(img).astype(np.float32)
                    current_mean = img_array.mean()
                    target_mean = 17.0
                    if current_mean > 0:
                        scale_factor = target_mean / current_mean
                        img_array = img_array * scale_factor
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img_array)
            else:
                # lsmi_utils not found - PIL cannot give us true raw sensor data
                # PIL applies automatic processing, so this is a fallback only
                print("Warning: lsmi_utils not found. PIL cannot provide true raw sensor data.")
                print("  PIL applies automatic white balance and processing.")
                print("  For true raw sensor format, ensure rawpy and lsmi_utils are available.")
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32)
                current_mean = img_array.mean()
                target_mean = 17.0
                if current_mean > 0:
                    scale_factor = target_mean / current_mean
                    img_array = img_array * scale_factor
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
        except Exception as e:
            print(f"Warning: Could not process NEF, falling back to PIL with darkening: {e}")
            img = Image.open(img_path).convert('RGB')
            # Make it dark to match training data
            img_array = np.array(img).astype(np.float32)
            current_mean = img_array.mean()
            target_mean = 17.0
            if current_mean > 0:
                scale_factor = target_mean / current_mean
                img_array = img_array * scale_factor
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
    else:
        # Standard image formats
        img = Image.open(img_path).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


def get_random_examples(dataset, examples_per_class=1):
    """Get random examples from each class."""
    examples = {class_idx: [] for class_idx in range(NUM_CLASSES)}
    
    for idx, (img, label) in enumerate(dataset):
        true_lbl = int(label)
        if len(examples[true_lbl]) < examples_per_class * 10:
            examples[true_lbl].append((img, true_lbl))
    
    selected = []
    for class_idx in range(NUM_CLASSES):
        if examples[class_idx]:
            selected.extend(
                random.sample(examples[class_idx], 
                            min(examples_per_class, len(examples[class_idx])))
            )
    
    return selected


def create_cam_instance(cam_method, model_wrapper, target_layers, model, model_type):
    """Create CAM instance based on method name."""
    cam_class = CAM_METHODS[cam_method.lower()]
    if cam_class is None:
        raise ValueError(f"Unknown CAM method: {cam_method}")
    return cam_class(model=model_wrapper, target_layers=target_layers)


def generate_heatmaps(model_type, cam_method, layer_name, 
                      single_image_path=None, random_mode=False, seed=42, multi_illuminant=False):
    """Main visualization function."""
    # Get model path from hardcoded paths
    model_path = MODEL_PATHS[model_type]
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load model
    print(f"\nLoading {model_type} model from {model_path}...")
    model_class = MODELS[model_type]
    model = model_class().to(DEVICE)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Get available layers
    available_layers = get_available_layers(model, model_type)
    layer_dict = {name: layer for name, layer in available_layers}
    
    if layer_name not in layer_dict:
        print(f"Error: Layer '{layer_name}' not found!")
        print(f"Available layers: {list(layer_dict.keys())}")
        return
    
    target_layer = layer_dict[layer_name]
    print(f"Using layer: {layer_name}")
    
    # Create CAM instance
    print(f"Initializing {cam_method.upper()}...")
    model_wrapper = ModelWrapper(model, model_type)
    
    try:
        cam = create_cam_instance(cam_method, model_wrapper, [target_layer], model, model_type)
    except Exception as e:
        print(f"Error creating CAM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    mean_np = np.array(MEAN, dtype=np.float32)
    std_np = np.array(STD, dtype=np.float32)
    
    # Load datasets for label names
    _, _, _, label_names = get_datasets()
    
    # Prepare images
    if single_image_path:
        # Single image mode
        print(f"Processing single image: {single_image_path}")
        img_tensor, _ = load_image_from_path(single_image_path)
        img_tensor = img_tensor.to(DEVICE)
        
        with torch.no_grad():
            if model_type == 'confidence':
                outputs, _ = model(img_tensor)
            else:
                # standard and paper models return logits directly
                outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred_label = outputs.argmax(dim=1)[0].item()
        
        images_to_process = [(img_tensor, None, pred_label)]
        
    elif random_mode:
        # Random 5 images mode
        print("Selecting random examples (one per class)...")
        train_dataset, val_dataset, test_dataset, _ = get_datasets()
        selected = get_random_examples(test_dataset, examples_per_class=1)
        
        images_to_process = []
        for img_tensor, true_label in selected:
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if model_type == 'confidence':
                    outputs, _ = model(img_tensor)
                else:
                    outputs = model(img_tensor)
                pred_label = outputs.argmax(dim=1)[0].item()
            images_to_process.append((img_tensor, true_label, pred_label))
    else:
        print("Error: Must specify either --single-image or --random")
        return
    
    # Load ground truth masks if multi-illuminant mode
    gt_masks = None
    if multi_illuminant and single_image_path:
        # Try to load mask for LSMI_Test_Package images
        img_basename = os.path.splitext(os.path.basename(single_image_path))[0]
        mask_path = os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package", "masks", f"{img_basename}_mask.npy")
        print(f"\nAttempting to load ground truth mask from: {mask_path}")
        if os.path.exists(mask_path):
            try:
                import sys
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data", "LSMI_Test_Package"))
                from lsmi_utils import load_mask
                # Load mask and resize to match image size (224x224 after transform)
                gt_masks = load_mask(mask_path, target_shape=(IMG_SIZE, IMG_SIZE))
                print(f"✓ Loaded ground truth mask: shape={gt_masks.shape}")
            except ImportError as e:
                print(f"Warning: Could not import lsmi_utils: {e}")
                # Try loading mask directly with numpy
                try:
                    gt_masks = np.load(mask_path)
                    # Resize if needed
                    if gt_masks.shape[:2] != (IMG_SIZE, IMG_SIZE):
                        import cv2
                        gt_masks = cv2.resize(gt_masks, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                    print(f"✓ Loaded ground truth mask directly: shape={gt_masks.shape}")
                except Exception as e2:
                    print(f"Warning: Could not load ground truth mask: {e2}")
                    gt_masks = None
            except Exception as e:
                print(f"Warning: Could not load ground truth mask: {e}")
                import traceback
                traceback.print_exc()
                gt_masks = None
        else:
            print(f"Warning: Mask file not found at {mask_path}")
            print(f"  Image basename: {img_basename}")
            print(f"  Expected mask: {img_basename}_mask.npy")
    
    # Process images
    processed = []
    print(f"\nGenerating heatmaps for {len(images_to_process)} image(s)...")
    
    for idx, (img_tensor, true_label, pred_label) in enumerate(images_to_process):
        rgb_image = tensor_to_rgb(img_tensor[0], mean_np, std_np)
        
        if true_label is not None:
            true_name = label_names[true_label]
        else:
            true_name = "Unknown"
        pred_name = label_names[pred_label]
        
        # Get probabilities
        with torch.no_grad():
            if model_type == 'confidence':
                outputs, _ = model(img_tensor)
            else:
                # standard and paper models return logits directly
                outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Generate CAM for all classes
        cams = []
        for class_idx in range(NUM_CLASSES):
            targets = [ClassifierOutputTarget(class_idx)]
            
            try:
                if cam_method in ['gradcam', 'gradcam++']:
                    with torch.enable_grad():
                        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
                else:
                    with torch.no_grad():
                        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
                heatmap = grayscale_cam
                
                # Force min-max normalization to match apply_ccm_with_cam.py
                # This ensures even weak activations are visualized with full dynamic range
                mn, mx = heatmap.min(), heatmap.max()
                if mx - mn > 1e-8:
                    heatmap = (heatmap - mn) / (mx - mn)
                
                # Add prob to title instead of dimming the heatmap
                prob = probs[class_idx].item()
                
                overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)
                cams.append((overlay, label_names[class_idx], prob))
                
            except Exception as e:
                print(f"Error generating CAM for class {class_idx}: {e}")
                overlay = np.zeros_like(rgb_image)
                cams.append((overlay, label_names[class_idx], 0.0))
        
        processed.append({
            "rgb": rgb_image,
            "cams": cams,
            "true": true_name,
            "pred": pred_name
        })
    
    # Create visualization grid
    # If multi-illuminant mode, add columns for ground truth masks
    rows = len(processed)
    mask_cols = NUM_CLASSES if (multi_illuminant and gt_masks is not None) else 0
    cols = 1 + NUM_CLASSES + mask_cols  # Original + CAM classes + GT masks (if multi)
    
    print(f"\nVisualization layout: {rows} row(s) x {cols} column(s)")
    print(f"  - Column 0: Original image")
    print(f"  - Columns 1-{NUM_CLASSES}: CAM visualizations ({NUM_CLASSES} classes)")
    if mask_cols > 0:
        print(f"  - Columns {NUM_CLASSES+1}-{cols-1}: Ground truth masks ({mask_cols} classes)")
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, item in enumerate(processed):
        col_idx = 0
        
        # Original image (always first column)
        axes[i, col_idx].imshow(item['rgb'])
        axes[i, col_idx].set_title(f"Original\nTrue: {item['true']}\nPred: {item['pred']}", fontsize=10, weight='bold')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # CAM for each class
        for j, (overlay, class_name, prob) in enumerate(item['cams']):
            ax = axes[i, col_idx]
            ax.imshow(overlay)
            
            title = f"{cam_method.upper()}\n{class_name}\n{prob:.1%}"
            if class_name == item['pred']:
                title += "\n(PREDICTED)"
                color = 'red'
                weight = 'bold'
            else:
                color = 'black'
                weight = 'normal'
            
            ax.set_title(title, fontsize=10, color=color, weight=weight)
            ax.axis('off')
            col_idx += 1
        
        # Ground truth masks (if multi-illuminant mode)
        if multi_illuminant and gt_masks is not None:
            # Map label names to mask channel order: ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
            mask_channel_order = ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
            # Convert RGB image to uint8 for cv2 processing
            rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
            
            for j, class_name in enumerate(label_names):
                ax = axes[i, col_idx]
                # Find corresponding mask channel
                if class_name in mask_channel_order:
                    mask_ch_idx = mask_channel_order.index(class_name)
                    mask_channel = gt_masks[:, :, mask_ch_idx]
                    # Normalize mask to 0-1 for display
                    if mask_channel.max() > 0:
                        mask_channel_norm = mask_channel / mask_channel.max()
                    else:
                        mask_channel_norm = mask_channel
                    
                    # Use JET colormap like in the evaluation notebook
                    try:
                        import cv2
                        # Apply JET colormap to mask (0-255 range)
                        mask_colored = cv2.applyColorMap(
                            np.uint8(255 * mask_channel_norm), 
                            cv2.COLORMAP_JET
                        )
                        # Convert BGR to RGB for matplotlib
                        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
                        # Overlay on original image (50/50 blend like notebook)
                        mask_overlay = (rgb_image_uint8 * 0.5 + mask_colored * 0.5).astype(np.uint8)
                        ax.imshow(mask_overlay)
                    except ImportError:
                        # Fallback if cv2 not available
                        mask_overlay = rgb_image.copy()
                        mask_colored = np.zeros_like(rgb_image)
                        mask_colored[:, :, 0] = mask_channel_norm  # Red channel
                        mask_overlay = mask_overlay * 0.6 + mask_colored * 0.4
                        mask_overlay = np.clip(mask_overlay, 0, 1)
                        ax.imshow(mask_overlay)
                    
                    ax.set_title(f"GT Mask\n{class_name}", fontsize=10, color='green', weight='bold')
                else:
                    ax.imshow(rgb_image)
                    ax.set_title(f"GT Mask\n{class_name}\n(N/A)", fontsize=10)
                ax.axis('off')
                col_idx += 1
    
    plt.tight_layout()
    
    # Save in visualizations/cams folder
    cams_dir = os.path.join(VISUALIZATIONS_DIR, "cams")
    os.makedirs(cams_dir, exist_ok=True)
    if single_image_path:
        img_name = os.path.splitext(os.path.basename(single_image_path))[0]
        save_path = os.path.join(cams_dir, 
                                f"{cam_method}_{model_type}_{layer_name}_{img_name}.png")
    else:
        save_path = os.path.join(cams_dir,
                                f"{cam_method}_{model_type}_{layer_name}_random5.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization to: {save_path}")


def interactive_mode():
    """Interactive mode for selecting model, layer, and CAM method."""
    print("\n" + "="*60)
    print("INTERACTIVE CAM VISUALIZATION")
    print("="*60)
    
    # Select model type
    print("\nAvailable models:")
    for i, name in enumerate(MODELS.keys(), 1):
        print(f"  {i}. {name}")
    model_choice = input(f"\nSelect model (1-{len(MODELS)}): ").strip()
    model_type = list(MODELS.keys())[int(model_choice) - 1]
    
    # Select model type
    print("\nAvailable models:")
    for i, name in enumerate(MODEL_PATHS.keys(), 1):
        print(f"  {i}. {name}")
    model_choice = input(f"\nSelect model (1-{len(MODEL_PATHS)}): ").strip()
    model_type = list(MODEL_PATHS.keys())[int(model_choice) - 1]
    model_path = MODEL_PATHS[model_type]
    print(f"Using model: {model_type} ({model_path})")
    
    # Load model to get layers
    print(f"\nLoading model to inspect layers...")
    model_class = MODELS[model_type]
    model = model_class().to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: Model file not found. Showing available layers anyway.")
    
    available_layers = get_available_layers(model, model_type)
    print("\nAvailable layers:")
    for i, (name, _) in enumerate(available_layers, 1):
        print(f"  {i}. {name}")
    layer_choice = input("\nSelect layer (1-5): ").strip()
    layer_name = available_layers[int(layer_choice) - 1][0]
    
    # Select CAM method
    print("\nAvailable CAM methods:")
    for i, name in enumerate(CAM_METHODS.keys(), 1):
        print(f"  {i}. {name}")
    cam_choice = input("\nSelect CAM method (1-3): ").strip()
    cam_method = list(CAM_METHODS.keys())[int(cam_choice) - 1]
    
    # Select mode
    print("\nVisualization mode:")
    print("  1. Single image (provide path)")
    print("  2. Random 5 images (one per class)")
    mode_choice = input("\nSelect mode (1-2): ").strip()
    
    if mode_choice == '1':
        img_path = input("Image path: ").strip()
        multi_choice = input("Multi-illuminant mode (show GT masks)? [y/N]: ").strip().lower()
        multi_mode = multi_choice == 'y'
        generate_heatmaps(model_type, cam_method, layer_name,
                         single_image_path=img_path, multi_illuminant=multi_mode)
    else:
        generate_heatmaps(model_type, cam_method, layer_name,
                         random_mode=True, multi_illuminant=False)


def main():
    parser = argparse.ArgumentParser(description='Unified CAM Visualization Tool')
    parser.add_argument('--model', type=str, choices=list(MODEL_PATHS.keys()), default='standard',
                       help='Model name: standard, confidence, paper, or illumicam3 (default: standard)')
    parser.add_argument('--cam', type=str, choices=list(CAM_METHODS.keys()),
                       help='CAM method to use')
    parser.add_argument('--layer', type=str, help='Layer name (e.g., conv5)')
    parser.add_argument('--single-image', type=str, help='Path to single image')
    parser.add_argument('--random', action='store_true', help='Use random 5 images')
    parser.add_argument('--multi', action='store_true', help='Multi-illuminant mode: show ground truth masks (for LSMI_Test_Package images)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        if not all([args.cam, args.layer]):
            print("Error: Must provide --cam and --layer")
            print("Or use --interactive for guided mode")
            return
        
        if not args.single_image and not args.random:
            print("Error: Must specify either --single-image or --random")
            return
        
        generate_heatmaps(
            args.model, args.cam, args.layer,
            single_image_path=args.single_image if args.single_image else None,
            random_mode=args.random,
            multi_illuminant=args.multi
        )


if __name__ == "__main__":
    main()

