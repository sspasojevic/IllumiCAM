import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ColorConstancyCNN(nn.Module):
    def __init__(self, K, pretrained=True):
        """
        K: The number of illuminant clusters (output classes).
        pretrained: Whether to use ImageNet pretrained weights.
        """
        super(ColorConstancyCNN, self).__init__()
        
        # Load Pretrained AlexNet (Closest architecture to the paper's custom model)
        # The paper describes a 5-conv layer network very similar to AlexNet/CaffeNet
        if pretrained:
            print("Loading ImageNet pretrained AlexNet weights...")
            self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.alexnet = models.alexnet(weights=None)

        # Feature extractor (Conv1 - Conv5 + MaxPool)
        self.features = self.alexnet.features
        
        # We replace the classifier to match the paper's specific FC structure
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096), # AlexNet uses 6x6 adaptive pooling
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, K), # Replace 1000 with K classes
        )
        
        # Initialize the new classifier layers
        self._initialize_classifier()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def _initialize_classifier(self):
        nn.init.normal_(self.classifier[6].weight, 0, 0.01)
        nn.init.constant_(self.classifier[6].bias, 0)

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
