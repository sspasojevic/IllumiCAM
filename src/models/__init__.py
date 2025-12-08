"""
Model implementations for illuminant estimation.
"""

from .model import IlluminantCNN, count_parameters
from .model_confidence import ConfidenceWeightedCNN
from .model_paper import ColorConstancyCNN
from .model_illumicam3 import IllumiCam3

__all__ = ['IlluminantCNN', 'ConfidenceWeightedCNN', 'ColorConstancyCNN', 'IllumiCam3', 'count_parameters']

