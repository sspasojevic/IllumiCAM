"""
Confidence-Weighted Pooling CNN for illuminant estimation.
Based on "Convolutional Color Constancy" (FC4) concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hardcoded configuration
NUM_CLASSES = 5
DROPOUT_RATE = 0.25


class ConfidenceWeightedCNN(nn.Module):
    """
    CNN with Confidence-Weighted Pooling.
    
    Instead of Max or Avg pooling, this network learns a confidence map
    to weight the spatial features before aggregation.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(ConfidenceWeightedCNN, self).__init__()

        # Shared Feature Extractor (Same as original)
        # Conv Block 1: Conv(32, 10x10) -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Conv(64, 7x7) -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Conv(96, 5x5) -> BN -> ReLU
        self.conv3 = nn.Conv2d(64, 96, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(96)

        # Conv Block 4: Conv(128, 5x5) -> BN -> ReLU
        self.conv4 = nn.Conv2d(96, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        # Conv Block 5: Conv(256, 3x3) -> BN -> ReLU
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        
        # --- BRANCHING POINT ---
        
        # 1. Confidence Branch: Outputs a 1-channel mask (H x W)
        # We use a 1x1 conv to collapse 256 channels to 1
        self.confidence_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # 2. Fully Connected Layers (applied AFTER pooling)
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x))) # Shape: [B, 256, H, W]
        
        # --- CONFIDENCE POOLING ---
        
        # 1. Compute Confidence Map
        # Shape: [B, 1, H, W]
        confidence = self.confidence_conv(x)
        
        # 2. Normalize confidence (softmax over spatial dimensions H*W)
        # This ensures weights sum to 1 for each image
        B, C, H, W = x.shape
        confidence_flat = confidence.view(B, 1, -1) # [B, 1, H*W]
        weights = F.softmax(confidence_flat, dim=2) # [B, 1, H*W]
        
        # 3. Reshape weights back to spatial for later visualization if needed
        spatial_weights = weights.view(B, 1, H, W)
        
        # 4. Weighted Pooling
        # Multiply features by weights and sum over spatial dimensions
        x_flat = x.view(B, C, -1) # [B, 256, H*W]
        
        # [B, 256, H*W] * [B, 1, H*W] -> [B, 256, H*W] -> sum -> [B, 256]
        weighted_features = torch.sum(x_flat * weights, dim=2)
        
        # --- CLASSIFICATION ---
        x = self.relu(self.fc1(weighted_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, spatial_weights

    def get_confidence_map(self, x):
        """Helper to extract just the confidence map for visualization."""
        # Run forward pass up to confidence
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        confidence = self.confidence_conv(x)
        return confidence


