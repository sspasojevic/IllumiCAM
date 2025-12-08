"""
Custom CNN model for illuminant estimation.
"""

import torch
import torch.nn as nn

# Hardcoded configuration
NUM_CLASSES = 5
DROPOUT_RATE = 0.25


class IlluminantCNN(nn.Module):
    """
    Custom CNN for illuminant classification.
    
    Architecture:
    - Conv Block 1: Conv(32, 10x10) -> BN -> ReLU -> MaxPool
    - Conv Block 2: Conv(64, 7x7) -> BN -> ReLU -> MaxPool
    - Conv Block 3: Conv(96, 5x5) -> BN -> ReLU
    - Conv Block 4: Conv(128, 5x5) -> BN -> ReLU
    - Conv Block 5: Conv(256, 3x3) -> BN -> ReLU
    - Global Max Pooling
    - FC1: 256 -> 1024 -> ReLU -> Dropout
    - FC2: 1024 -> num_classes
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(IlluminantCNN, self).__init__()

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

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Fully Connected
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))

        # Block 5
        x = self.relu(self.bn5(self.conv5(x)))

        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

