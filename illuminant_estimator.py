"""
Illuminant estimation using continuous estimation from discrete classification.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from config import (
    DEVICE, BEST_MODEL_PATH_VAL, CLUSTER_CENTERS_PATH, 
    NUM_CLASSES, MEAN, STD, IMG_SIZE, VISUALIZATIONS_DIR
)
from model import IlluminantCNN
from data_loader import get_datasets, get_dataloaders, get_transforms


class IlluminantEstimator:
    """
    Estimates continuous illuminant chromaticity using:
    e = Σ(P(y=i|x) * μ_i)
    where μ_i are cluster centers for each class.
    """
    
    def __init__(self, model, cluster_centers, class_names, device='cuda'):
        """
        Initialize the illuminant estimator.
        
        Args:
            model: Trained IlluminantCNN model
            cluster_centers: Dictionary mapping class names to RGB chromaticity values
            class_names: List of class names
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.K = len(class_names)
        
        # Build cluster center matrix [K, 3]
        self.mu = np.zeros((self.K, 3), dtype=np.float32)
        for i, name in enumerate(class_names):
            self.mu[i] = cluster_centers[name]
        
        self.mu_tensor = torch.from_numpy(self.mu).to(device)
        
        print("\nCluster centers loaded:")
        for i, name in enumerate(class_names):
            print(f"  {name}: μ = [{self.mu[i,0]:.4f}, {self.mu[i,1]:.4f}, {self.mu[i,2]:.4f}]")
    
    def estimate(self, image_tensor):
        """
        Single image estimation.
        
        Args:
            image_tensor: Image tensor (C, H, W) or (1, C, H, W)
        
        Returns:
            Dictionary with:
                - illuminant: Continuous RGB illuminant estimate
                - probabilities: Class probabilities
                - predicted_class: Predicted class index
                - discrete_illuminant: Discrete illuminant (argmax)
        """
        self.model.eval()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Weighted average: e = Σ(p_i * μ_i)
        illuminant = torch.matmul(probs, self.mu_tensor)
        
        return {
            'illuminant': illuminant.cpu().numpy()[0],
            'probabilities': probs.cpu().numpy()[0],
            'predicted_class': probs.argmax(dim=1).item(),
            'discrete_illuminant': self.mu[probs.argmax(dim=1).item()]
        }
    
    def estimate_batch(self, image_batch):
        """
        Batch estimation.
        
        Args:
            image_batch: Batch of image tensors (B, C, H, W)
        
        Returns:
            Dictionary with batch results
        """
        self.model.eval()
        image_batch = image_batch.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_batch)
            probs = F.softmax(logits, dim=1)
        
        illuminants = torch.matmul(probs, self.mu_tensor)
        pred_classes = probs.argmax(dim=1)
        discrete_illuminants = self.mu[pred_classes.cpu().numpy()]
        
        return {
            'illuminants': illuminants.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'predicted_classes': pred_classes.cpu().numpy(),
            'discrete_illuminants': discrete_illuminants
        }


def load_estimator(model_path=BEST_MODEL_PATH_VAL, cluster_centers_path=CLUSTER_CENTERS_PATH):
    """
    Load model and cluster centers, create estimator.
    
    Args:
        model_path: Path to saved model
        cluster_centers_path: Path to cluster centers file
    
    Returns:
        estimator: IlluminantEstimator instance
        class_names: List of class names
    """
    # Load model
    model = IlluminantCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Load cluster centers
    cluster_centers = np.load(cluster_centers_path, allow_pickle=True).item()
    print(f"Loaded cluster centers from: {cluster_centers_path}")
    
    # Get class names from dataset
    train_dataset, _, _, class_names = get_datasets()
    
    # Create estimator
    estimator = IlluminantEstimator(model, cluster_centers, class_names, DEVICE)
    
    return estimator, class_names


def estimate_test_set(estimator, class_names, batch_size=32):
    """
    Run estimation on test set and generate visualizations.
    
    Args:
        estimator: IlluminantEstimator instance
        class_names: List of class names
        batch_size: Batch size for processing
    """
    print("\n" + "="*60)
    print("RUNNING ILLUMINANT ESTIMATION ON TEST SET")
    print("="*60)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset, _ = get_datasets()
    
    # Create test loader
    _, _, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=batch_size
    )
    
    # Run estimation
    all_illuminants = []
    all_discrete_illuminants = []
    all_probs = []
    all_preds = []
    all_true_labels = []
    
    for images, labels in tqdm(test_loader, desc="Estimating"):
        results = estimator.estimate_batch(images)
        
        all_illuminants.append(results['illuminants'])
        all_discrete_illuminants.append(results['discrete_illuminants'])
        all_probs.append(results['probabilities'])
        all_preds.append(results['predicted_classes'])
        all_true_labels.append(labels.numpy())
    
    # Concatenate all results
    all_illuminants = np.concatenate(all_illuminants, axis=0)
    all_discrete_illuminants = np.concatenate(all_discrete_illuminants, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    
    # Print results
    print("\n" + "="*60)
    print("ESTIMATION RESULTS")
    print("="*60)
    
    accuracy = (all_preds == all_true_labels).mean()
    print(f"\nClassification Accuracy: {accuracy:.4f}")
    
    print(f"\nEstimated Illuminants Shape: {all_illuminants.shape}")
    print(f"\nContinuous Illuminant Statistics:")
    print(f"  R: mean={all_illuminants[:,0].mean():.4f}, std={all_illuminants[:,0].std():.4f}, "
          f"min={all_illuminants[:,0].min():.4f}, max={all_illuminants[:,0].max():.4f}")
    print(f"  G: mean={all_illuminants[:,1].mean():.4f}, std={all_illuminants[:,1].std():.4f}, "
          f"min={all_illuminants[:,1].min():.4f}, max={all_illuminants[:,1].max():.4f}")
    print(f"  B: mean={all_illuminants[:,2].mean():.4f}, std={all_illuminants[:,2].std():.4f}, "
          f"min={all_illuminants[:,2].min():.4f}, max={all_illuminants[:,2].max():.4f}")
    
    # Per-class statistics
    print("\nPer-Class Illuminant Estimates:")
    for i, name in enumerate(class_names):
        mask = all_true_labels == i
        if mask.sum() > 0:
            class_illuminants = all_illuminants[mask]
            print(f"  {name:12s}: n={mask.sum():4d}, "
                  f"mean=[{class_illuminants[:,0].mean():.4f}, "
                  f"{class_illuminants[:,1].mean():.4f}, "
                  f"{class_illuminants[:,2].mean():.4f}]")
    
    # Generate visualizations
    visualize_estimation_results(
        all_illuminants, all_discrete_illuminants, all_probs, 
        all_preds, all_true_labels, estimator, class_names
    )
    
    # Visualize individual examples
    visualize_individual_examples(estimator, class_names, test_dataset)
    
    # Save results
    results_dict = {
        'illuminants': all_illuminants,
        'discrete_illuminants': all_discrete_illuminants,
        'probabilities': all_probs,
        'predicted_classes': all_preds,
        'true_labels': all_true_labels,
        'class_names': class_names,
        'cluster_centers': estimator.mu
    }
    
    save_path = "illuminant_estimation_results.npy"
    np.save(save_path, results_dict)
    print(f"\nResults saved to {save_path}")
    
    return results_dict


def visualize_estimation_results(all_illuminants, all_discrete_illuminants, all_probs,
                                 all_preds, all_true_labels, estimator, class_names):
    """Generate visualization plots for estimation results."""
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Chromaticity scatter (R vs G)
    ax = axes[0, 0]
    scatter_colors = plt.cm.tab10(all_true_labels / (num_classes - 1))
    ax.scatter(all_illuminants[:, 0], all_illuminants[:, 1], c=scatter_colors, alpha=0.4, s=15)
    for i, name in enumerate(class_names):
        ax.scatter(estimator.mu[i, 0], estimator.mu[i, 1],
                   marker='X', s=300, edgecolors='black', linewidth=2, label=f"μ_{name}")
    ax.set_xlabel("R chromaticity")
    ax.set_ylabel("G chromaticity")
    ax.set_title("Continuous Illuminant Estimates (R vs G)")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Chromaticity scatter (R vs B)
    ax = axes[0, 1]
    ax.scatter(all_illuminants[:, 0], all_illuminants[:, 2], c=scatter_colors, alpha=0.4, s=15)
    for i, name in enumerate(class_names):
        ax.scatter(estimator.mu[i, 0], estimator.mu[i, 2],
                   marker='X', s=300, edgecolors='black', linewidth=2)
    ax.set_xlabel("R chromaticity")
    ax.set_ylabel("B chromaticity")
    ax.set_title("Continuous Illuminant Estimates (R vs B)")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Discrete vs Continuous comparison
    ax = axes[0, 2]
    ax.scatter(all_discrete_illuminants[:, 0], all_illuminants[:, 0], alpha=0.3, s=10, label='R')
    ax.scatter(all_discrete_illuminants[:, 1], all_illuminants[:, 1], alpha=0.3, s=10, label='G')
    ax.scatter(all_discrete_illuminants[:, 2], all_illuminants[:, 2], alpha=0.3, s=10, label='B')
    lims = [0.25, 0.45]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel("Discrete (argmax μ)")
    ax.set_ylabel("Continuous (weighted)")
    ax.set_title("Discrete vs Continuous Estimation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Probability distribution samples
    ax = axes[1, 0]
    sample_idx = np.random.choice(len(all_probs), min(15, len(all_probs)), replace=False)
    x_pos = np.arange(num_classes)
    width = 0.05
    for j, idx in enumerate(sample_idx):
        ax.bar(x_pos + j*width, all_probs[idx], width, alpha=0.7)
    ax.set_xticks(x_pos + width * len(sample_idx) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel("Probability")
    ax.set_title("Sample Probability Distributions")
    
    # Plot 5: Illuminant distribution histograms
    ax = axes[1, 1]
    ax.hist(all_illuminants[:, 0], bins=50, alpha=0.5, label='R', color='red')
    ax.hist(all_illuminants[:, 1], bins=50, alpha=0.5, label='G', color='green')
    ax.hist(all_illuminants[:, 2], bins=50, alpha=0.5, label='B', color='blue')
    ax.set_xlabel("Chromaticity value")
    ax.set_ylabel("Count")
    ax.set_title("Illuminant Distribution")
    ax.legend()
    
    # Plot 6: Per-class box plot
    ax = axes[1, 2]
    data_for_box = []
    labels_for_box = []
    for i, name in enumerate(class_names):
        mask = all_true_labels == i
        if mask.sum() > 0:
            data_for_box.append(all_illuminants[mask, 0])
            labels_for_box.append(name)
    ax.boxplot(data_for_box, tick_labels=labels_for_box)
    ax.set_ylabel("R chromaticity")
    ax.set_title("R Chromaticity by Class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    save_path = os.path.join(VISUALIZATIONS_DIR, "illuminant_estimation_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_individual_examples(estimator, class_names, test_dataset, num_examples_per_class=2):
    """
    Visualize individual examples from each class showing probability contributions.
    
    Args:
        estimator: IlluminantEstimator instance
        class_names: List of class names
        test_dataset: Test dataset
        num_examples_per_class: Number of examples to show per class
    """
    print("\n" + "="*60)
    print("INDIVIDUAL EXAMPLES")
    print("="*60)
    
    def tensor_to_rgb(img_tensor):
        """Denormalize tensor to RGB [0,1]."""
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.array(STD) * img_np + np.array(MEAN)
        img_np = np.clip(img_np, 0.0, 1.0)
        return img_np
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_examples_per_class * 2,
                              figsize=(4 * num_examples_per_class * 2, 4 * num_classes))
    
    # Handle case where num_classes = 1
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for class_idx, class_name in enumerate(class_names):
        # Get images from this class
        class_indices = [i for i, (_, label) in enumerate(test_dataset.samples) if label == class_idx]
        
        for ex_idx in range(num_examples_per_class):
            if ex_idx >= len(class_indices):
                # Fill empty slots if not enough examples
                axes[class_idx, ex_idx * 2].axis('off')
                axes[class_idx, ex_idx * 2 + 1].axis('off')
                continue
            
            img_idx = class_indices[ex_idx]
            img_tensor, true_label = test_dataset[img_idx]
            
            # Estimate
            result = estimator.estimate(img_tensor)
            
            # Original image
            ax_img = axes[class_idx, ex_idx * 2]
            rgb_img = tensor_to_rgb(img_tensor)
            ax_img.imshow(rgb_img)
            ax_img.set_title(f"True: {class_name}", fontsize=10)
            ax_img.axis('off')
            
            # Probability bar chart
            ax_bar = axes[class_idx, ex_idx * 2 + 1]
            colors = ['red' if i == result['predicted_class'] else 'steelblue'
                      for i in range(num_classes)]
            ax_bar.barh(class_names, result['probabilities'], color=colors)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_title(f"e=[{result['illuminant'][0]:.3f}, "
                            f"{result['illuminant'][1]:.3f}, "
                            f"{result['illuminant'][2]:.3f}]", fontsize=9)
            
            # Add illuminant color patch
            ill_color = np.clip(result['illuminant'] * 3, 0, 1)  # Scale for visibility
            ax_bar.axvline(x=0.95, color=ill_color, linewidth=15)
    
    plt.tight_layout()
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    save_path = os.path.join(VISUALIZATIONS_DIR, "illuminant_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved individual examples visualization to {save_path}")


def estimate_single_image(image_path, estimator, class_names):
    """
    Convenience function to estimate illuminant for any image path.
    
    Args:
        image_path: Path to image file
        estimator: IlluminantEstimator instance
        class_names: List of class names
    
    Returns:
        Dictionary with estimation results
    """
    _, val_test_transform = get_transforms()
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_test_transform(img)
    
    result = estimator.estimate(img_tensor)
    
    print(f"\nImage: {image_path}")
    print(f"Predicted class: {class_names[result['predicted_class']]}")
    print(f"Confidence: {result['probabilities'][result['predicted_class']]:.4f}")
    print(f"\nClass probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {result['probabilities'][i]:.4f}")
    print(f"\nEstimated illuminant (continuous): "
          f"[{result['illuminant'][0]:.4f}, {result['illuminant'][1]:.4f}, {result['illuminant'][2]:.4f}]")
    print(f"Discrete illuminant (argmax):     "
          f"[{result['discrete_illuminant'][0]:.4f}, {result['discrete_illuminant'][1]:.4f}, {result['discrete_illuminant'][2]:.4f}]")
    
    # Show image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[result['predicted_class']]}\n"
              f"Illuminant: [{result['illuminant'][0]:.3f}, {result['illuminant'][1]:.3f}, {result['illuminant'][2]:.3f}]")
    plt.axis('off')
    plt.show()
    
    return result


def main():
    """Main function for illuminant estimation."""
    estimator, class_names = load_estimator()
    estimate_test_set(estimator, class_names)


if __name__ == "__main__":
    main()

