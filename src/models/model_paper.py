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
        
        # The classifier in AlexNet is:
        # (0): Dropout(p=0.5, inplace=False)
        # (1): Linear(in_features=9216, out_features=4096, bias=True)
        # (2): ReLU(inplace=True)
        # (3): Dropout(p=0.5, inplace=False)
        # (4): Linear(in_features=4096, out_features=4096, bias=True)
        # (5): ReLU(inplace=True)
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        
        # We replace the classifier to match the paper's specific FC structure (if different)
        # or just modify the last layer. The paper specifies:
        # FC6 (4096) -> FC7 (4096) -> FC8 (K)
        # This matches AlexNet exactly, except for the last layer.
        
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
        # Initialize only the new classifier layers
        # Copy weights from pre-trained AlexNet classifier layers 1 and 4 if possible,
        # otherwise initialize randomly.
        
        # 1. Copy FC6 and FC7 weights from pretrained model to speed up convergence
        # self.classifier[1].weight.data = self.alexnet.classifier[1].weight.data
        # self.classifier[1].bias.data = self.alexnet.classifier[1].bias.data
        # self.classifier[4].weight.data = self.alexnet.classifier[4].weight.data
        # self.classifier[4].bias.data = self.alexnet.classifier[4].bias.data
        
        # For the final layer (K classes), initialize with Normal dist
        nn.init.normal_(self.classifier[6].weight, 0, 0.01)
        nn.init.constant_(self.classifier[6].bias, 0)

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
