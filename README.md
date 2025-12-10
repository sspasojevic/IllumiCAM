# Final Project

**Students:** Sara Spasojevic, Adnan Amir, and Ritik Bompilwar

**Class:** CS7180 - Advanced Computer Vision  

**Project:** Final Project - Grad-CAM Based Multi-Illuminant Detection and Spatial Localization for Selective White Balance Correction

**OS:** macOS (Apple Silicon M1/M2/M3) with MPS support  

## Project Overview

This project implements multiple CNN architectures for illuminant estimation. We train models to classify scene illuminants into five categories (Very Warm, Warm, Neutral, Cool, Very Cool) and use Class Activation Maps (CAM) to visualize spatial attention. The project includes spatially-aware color correction using CAM-guided white balance, evaluation on LSMI (Localized Spatially Mixed Illuminant) test images, and comparison of different CAM methods (GradCAM, GradCAM++, ScoreCAM).

## Dependencies

### Required Packages
```
torch>=2.0.0
torchvision>=0.15.0
numpy
opencv-python
rawpy
scipy
scikit-learn
matplotlib
seaborn
Pillow
tqdm
pandas
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Folder Structure

```
Final_Project/
├── config/
│   └── config.py                        # Centralized configuration
├── src/
│   ├── models/
│   │   ├── model.py                     # IlluminantCNN (standard model)
│   │   ├── model_confidence.py          # ConfidenceWeightedCNN (FC4-inspired)
│   │   ├── model_paper.py               # ColorConstancyCNN (AlexNet-based)
│   │   └── model_illumicam3.py          # IllumiCam3 (Global Average Pooling)
│   ├── display_scripts/
│   │   ├── visualize_cam.py             # CAM visualization tool
│   │   ├── visualize_gt_masks.py        # Ground truth mask visualization
│   │   └── correct_with_cam.py          # CAM-guided color correction
│   ├── data_manipulations/
│   │   ├── augment_split_data.py        # Dataset creation and splitting
│   │   └── generate_lsmi_masks.py       # LSMI ground truth mask generation
│   ├── utils.py                         # Core utilities
│   ├── data_loader.py                   # Data loading and transforms
│   ├── train.py                         # Training script
│   ├── evaluate_models.py               # Model evaluation on test set
│   └── evaluate_masks.py                # CAM evaluation against GT masks
├── Data/
│   ├── Nikon_D810/                      # Raw illuminant data
│   ├── LSMI/                            # LSMI test dataset (RAW)
│   │   ├── nikon/                       # Nikon subset of LSMI
│   │   └── masks/                       # Generated masks for all images
│   ├── LSMI_Test_Package/               # LSMI preprocessed
│   │   ├── images/                      # NEF test images
│   │   └── masks/                       # Ground truth masks (.npy)
│   └── info/                            # Camera CCMs and reference WPs
├── dataset/                             # Generated training dataset
│   ├── train/
│   ├── val/
│   └── test/
├── saved_models/                        # Trained model weights
│   ├── best_illuminant_cnn_val_8084.pth
│   ├── best_illuminant_cnn_confidence.pth
│   ├── best_paper_model.pth
│   └── best_illumicam3.pth
├── visualizations/                      # Generated visualizations
│   ├── cams/
│   ├── gt_masks/
│   ├── cam_correction/
│   └── evaluate/
├── cluster_centers.npy                  # Illuminant cluster centers
└── requirements.txt
```

## Usage

### 1. Dataset Preparation

#### Generate Training Dataset
```bash
# Process raw data, cluster illuminants, balance classes, and split
python src/data_manipulations/augment_split_data.py
```

#### Generate LSMI Ground Truth Masks
```bash
python src/data_manipulations/generate_lsmi_masks.py \
    --lsmi_root Data/LSMI/nikon \
    --output_dir Data/LSMI_Test_Package/masks
```

### 2. Training

#### Train IllumiCAM (Standard) Model
```bash
python src/train.py --model-type standard
```

#### Train Confidence Model (FC4-inspired)
```bash
python src/train.py --model-type confidence
```

#### Train Color Constancy Paper Model (AlexNet-based)
```bash
python src/train.py --model-type paper
```

#### Train IllumiCAM3 Model
```bash
python src/train.py --model-type illumicam3
```

### 3. Evaluation

#### Evaluate Model on Test Set
```bash
python src/evaluate_models.py \
    --model-type standard \
    --batch-size 256
```

#### Evaluate CAM vs Ground Truth Masks
```bash
# Single model-CAM combination
python src/evaluate_masks.py \
    --model standard \
    --cam gradcam \
    --output results.csv

# Comprehensive matrix evaluation (all combinations)
python src/evaluate_masks.py \
    --matrix \
    --output results_matrix.csv
```

### 4. Visualization

#### Visualize CAM on Images
```bash

# From LSMI NEF file
python src/display_scripts/visualize_cam.py \
    --model standard \
    --cam gradcam \
    --layer conv5 \
    --image Data/LSMI_Test_Package/images/Place101.nef
```

#### Visualize Ground Truth Masks
```bash
python src/display_scripts/visualize_gt_masks.py \
    --image Data/LSMI_Test_Package/images/Place101.nef
```

#### CAM-Guided Color Correction
```bash
python src/display_scripts/correct_with_cam.py \
    --image Data/LSMI_Test_Package/images/Place101.nef \
    --model standard \
    --cam gradcam \
    --layer conv5
```

## Model Architectures

### 1. Standard IlluminantCNN
Custom 5-layer CNN with BatchNorm, MaxPool, and Global Max Pooling.
- **Best Model:** `best_illuminant_cnn_val_8084.pth` (84.49% test accuracy)

### 2. Confidence-Weighted CNN
FC4-inspired model with learned spatial confidence weighting instead of fixed pooling.
- **Best Model:** `best_illuminant_cnn_confidence.pth`

### 3. Paper Model (ColorConstancyCNN)
AlexNet-based architecture from color constancy literature.
- **Best Model:** `best_paper_model.pth`

### 4. IllumiCam3
Custom CNN with Global Average Pooling for better CAM interpretability.
- **Best Model:** `best_illumicam3.pth`

## CAM Methods Supported

- **GradCAM:** Gradient-weighted Class Activation Mapping
- **GradCAM++:** Improved localization with weighted gradients
- **ScoreCAM:** Gradient-free activation mapping

## Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

### Spatial Metrics (CAM vs GT Masks)
- **IOU:** Intersection over Union
- **DICE:** DICE coefficient
- **MAE:** Mean Absolute Error

## Hardware
MacOS 15.1, MacBook Pro, Apple Silicon (MPS acceleration)

## References

- **FC4:** "Convolutional Color Constancy" by Barron (ICCV 2015)
- **GradCAM:** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" by Selvaraju et al. (ICCV 2017)
- **LSMI Dataset:** Localized Spatially Mixed Illuminant dataset
- **pytorch_grad_cam:** [GitHub Repository](https://github.com/jacobgil/pytorch-grad-cam)

## License

This project is for educational purposes as part of CS7180 Advanced Computer Vision course.
