"""
Sara Spasojevic, Adnan Amir, Ritik Bompilwar
CS7180 Final Project, Fall 2025
December 9, 2025

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
        """
        Initialize ConfidenceWeightedCNN.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """

        super(ConfidenceWeightedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 96, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Confidence branch
        self.confidence_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # Classification branch
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Tuple of (logits, spatial_weights) where logits is shape (batch_size, num_classes)
            and spatial_weights is shape (batch_size, 1, height, width)
        """
        
        # Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        # Confidence pooling
        confidence = self.confidence_conv(x)
        
        B, C, H, W = x.shape
        confidence_flat = confidence.view(B, 1, -1)
        weights = F.softmax(confidence_flat, dim=2)
        
        spatial_weights = weights.view(B, 1, H, W)
        
        x_flat = x.view(B, C, -1)
        weighted_features = torch.sum(x_flat * weights, dim=2)
        
        x = self.relu(self.fc1(weighted_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, spatial_weights

    def get_confidence_map(self, x):
        """
        Extract confidence map for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Confidence map tensor of shape (batch_size, 1, height, width)
        """

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        confidence = self.confidence_conv(x)
        return confidence


