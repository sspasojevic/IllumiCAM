"""
Grad-CAM visualization for illuminant estimation model.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import DEVICE, BEST_MODEL_PATH_VAL, MEAN, STD, NUM_CLASSES, VISUALIZATIONS_DIR
from model import IlluminantCNN
from data_loader import get_datasets, get_dataloaders


def tensor_to_rgb(img_tensor, mean, std):
    """Convert normalized tensor to RGB [0,1] for visualization."""
    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np.astype(np.float32)


def generate_gradcam_heatmaps(model_path=BEST_MODEL_PATH_VAL, examples_per_class=1, 
                              batch_size=256, seed=42):
    """
    Generate Grad-CAM heatmaps for examples from each class.
    
    Args:
        model_path: Path to the saved model
        examples_per_class: Number of examples to visualize per class
        batch_size: Batch size for data loading
        seed: Random seed for reproducibility
    """
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM HEATMAPS")
    print("="*60)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset, label_names = get_datasets()
    
    # Create test loader
    _, _, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=batch_size
    )
    
    # Load model
    model = IlluminantCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Setup GradCAM
    mean_np = np.array(MEAN, dtype=np.float32)
    std_np = np.array(STD, dtype=np.float32)
    target_layers = [model.conv5]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Collect examples from each class
    examples = {class_idx: [] for class_idx in range(NUM_CLASSES)}
    
    print("Collecting examples from test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            for i in range(images.size(0)):
                true_label = int(labels[i].item())
                
                # Collect examples for each class
                if len(examples[true_label]) < examples_per_class * 10:
                    examples[true_label].append((
                        images[i], 
                        true_label, 
                        int(preds[i].item())
                    ))
            
            # Check if we have enough examples from all classes
            if all(len(examples[class_idx]) >= examples_per_class 
                   for class_idx in range(NUM_CLASSES)):
                break
    
    # Randomly select examples
    selected_examples = []
    for class_idx in range(NUM_CLASSES):
        if len(examples[class_idx]) > 0:
            selected = random.sample(
                examples[class_idx], 
                min(examples_per_class, len(examples[class_idx]))
            )
            selected_examples.extend(selected)
    
    print(f"Selected {len(selected_examples)} examples (one from each class)")
    
    # Process selected examples
    processed_examples = []
    for input_tensor, true_label, pred_label in selected_examples:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        rgb_image = tensor_to_rgb(input_tensor[0], mean_np, std_np)
        true_name = label_names[true_label]
        pred_name = label_names[pred_label]
        
        # Generate GradCAM for all classes
        overlays_all_classes = []
        for class_idx in range(NUM_CLASSES):
            targets = [ClassifierOutputTarget(class_idx)]
            
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            
            overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            overlays_all_classes.append((overlay, label_names[class_idx]))
        
        processed_examples.append((rgb_image, overlays_all_classes, true_name, pred_name))
    
    # Plot grid: Original + 5 GradCAMs (one for each class)
    rows = len(processed_examples)
    if rows > 0:
        # 1 original + 5 GradCAMs = 6 columns
        fig, axes = plt.subplots(rows, 6, figsize=(18, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for idx, (rgb_image, overlays_all_classes, true_name, pred_name) in enumerate(processed_examples):
            # Original image
            axes[idx, 0].imshow(rgb_image)
            axes[idx, 0].set_title(f"Original\nTrue: {true_name}\nPred: {pred_name}", fontsize=10)
            axes[idx, 0].axis("off")
            
            # GradCAM for each class
            for class_idx, (overlay, class_name) in enumerate(overlays_all_classes):
                axes[idx, class_idx + 1].imshow(overlay)
                # Highlight if this is the predicted class
                title = f"Grad-CAM\n{class_name}"
                if class_name == pred_name:
                    title += "\n(PREDICTED)"
                axes[idx, class_idx + 1].set_title(
                    title, 
                    fontsize=10, 
                    color='red' if class_name == pred_name else 'black',
                    weight='bold' if class_name == pred_name else 'normal'
                )
                axes[idx, class_idx + 1].axis("off")
        
        plt.tight_layout()
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(VISUALIZATIONS_DIR, "gradcam_grid_all_classes.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Grad-CAM grid to {save_path}")


def main():
    """Main function for Grad-CAM visualization."""
    generate_gradcam_heatmaps()


if __name__ == "__main__":
    main()

