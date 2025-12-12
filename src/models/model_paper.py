"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

Color Constancy CNN paper model implementation.
Uses AlexNet architecture and weights with custom classifier layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ColorConstancyCNN(nn.Module):
    """
    Color Constancy CNN based on AlexNet architecture.
    
    Uses AlexNet features with custom classifier layers for illuminant estimation.
    """
    
    def __init__(self, K, pretrained=True):
        """
        Initialize Color Constancy CNN.
        
        Args:
            K: Number of illuminant clusters (output classes)
            pretrained: Whether to use ImageNet pretrained weights
        
        Returns:
            Initialized model
        """

        super(ColorConstancyCNN, self).__init__()
        
        # Load Pretrained AlexNet (Closest architecture to the paper's custom model)
        if pretrained:
            print("Loading ImageNet pretrained AlexNet weights...")
            self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.alexnet = models.alexnet(weights=None)

        # Feature extractor (Conv1 - Conv5 + MaxPool)
        self.features = self.alexnet.features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, K),
        )
        
        # Initialize the new classifier layers
        self._initialize_classifier()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, K)
        """

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def _initialize_classifier(self):
        """
        Initialize classifier layer weights.
        """

        nn.init.normal_(self.classifier[6].weight, 0, 0.01)
        nn.init.constant_(self.classifier[6].bias, 0)

def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
