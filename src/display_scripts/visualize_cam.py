"""
Unified CAM Visualization Tool
Supports: GradCAM, GradCAM++, ScoreCAM
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import all CAM methods
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import project modules
from src.models.model import IlluminantCNN
from src.models.model_confidence import ConfidenceWeightedCNN
from src.models.model_paper import ColorConstancyCNN
from src.models.model_illumicam3 import IllumiCam3
from src.data_loader import get_datasets, get_dataloaders
from src.lsmi_utils import process_raw_image

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
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")


# Model registry
MODELS = {
    'standard': IlluminantCNN,
    'confidence': ConfidenceWeightedCNN,
    'paper': lambda: ColorConstancyCNN(K=NUM_CLASSES, pretrained=False),
    'illumicam3': IllumiCam3
}

# Model paths
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
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    # Handle NEF files using process_raw_image from lsmi_utils
    if img_path.lower().endswith('.nef'):
        # Load raw sensor data with minimal processing (srgb=False for linear RGB)
        rgb_array = process_raw_image(img_path, srgb=False)
        img = Image.fromarray(rgb_array)
        img_tensor = transform(img).unsqueeze(0)
    else:
        # Standard image formats
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img


def create_cam_instance(cam_method, model_wrapper, target_layers, model, model_type):
    """Create CAM instance based on method name."""
    cam_class = CAM_METHODS[cam_method.lower()]
    if cam_class is None:
        raise ValueError(f"Unknown CAM method: {cam_method}")
    return cam_class(model=model_wrapper, target_layers=target_layers)


def generate_heatmaps(model_type, cam_method, layer_name, image_path):
    """Main visualization function."""
    # Get model path from hardcoded paths
    model_path = MODEL_PATHS[model_type]
    
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
    cam = create_cam_instance(cam_method, model_wrapper, [target_layer], model, model_type)
    
    mean_np = np.array(MEAN, dtype=np.float32)
    std_np = np.array(STD, dtype=np.float32)
    
    # Load datasets for label names
    _, _, _, label_names = get_datasets()
    
    # Process single image
    print(f"Processing image: {image_path}")
    img_tensor, _ = load_image_from_path(image_path)
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
            
            if cam_method in ['gradcam', 'gradcam++']:
                with torch.enable_grad():
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
            else:
                with torch.no_grad():
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
            heatmap = grayscale_cam
            
            # Force min-max normalization to match correct_with_cam.py
            # This ensures even weak activations are visualized with full dynamic range
            mn, mx = heatmap.min(), heatmap.max()
            if mx - mn > 1e-8:
                heatmap = (heatmap - mn) / (mx - mn)
            
            # Add prob to title instead of dimming the heatmap
            prob = probs[class_idx].item()
            
            overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)
            cams.append((overlay, label_names[class_idx], prob))
        
        processed.append({
            "rgb": rgb_image,
            "cams": cams,
            "true": true_name,
            "pred": pred_name
        })
    
    # Create visualization grid
    rows = len(processed)
    cols = 1 + NUM_CLASSES  # Original + CAM classes
    
    print(f"\nVisualization layout: {rows} row(s) x {cols} column(s)")
    print(f"  - Column 0: Original image")
    print(f"  - Columns 1-{NUM_CLASSES}: CAM visualizations ({NUM_CLASSES} classes)")
    
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
    
    plt.tight_layout()
    
    # Save in visualizations/cams folder
    cams_dir = os.path.join(VISUALIZATIONS_DIR, "cams")
    os.makedirs(cams_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(cams_dir, 
                            f"{cam_method}_{model_type}_{layer_name}_{img_name}.png")
    
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
    
    # Get image path
    img_path = input("Image path: ").strip()
    generate_heatmaps(model_type, cam_method, layer_name, img_path)


def main():
    parser = argparse.ArgumentParser(description='Unified CAM Visualization Tool')
    parser.add_argument('--model', type=str, choices=list(MODEL_PATHS.keys()), default='standard',
                       help='Model name: standard, confidence, paper, or illumicam3 (default: standard)')
    parser.add_argument('--cam', type=str, choices=list(CAM_METHODS.keys()),
                       help='CAM method to use')
    parser.add_argument('--layer', type=str, help='Layer name (e.g., conv5)')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        if not all([args.cam, args.layer]):
            print("Error: Must provide --cam and --layer")
            print("Or use --interactive for guided mode")
            return
        
        generate_heatmaps(args.model, args.cam, args.layer, args.image)


if __name__ == "__main__":
    main()

