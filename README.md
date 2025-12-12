# Final Project

**Students:** Sara Spasojevic, Adnan Amir, and Ritik Bompilwar

**Class:** CS7180 - Advanced Computer Vision  

**Project:** Final Project - Grad-CAM Based Multi-Illuminant Detection and Spatial Localization for Selective White Balance Correction

**OS:** macOS (Apple Silicon M1/M2/M3) with MPS support  

## Project Overview

This project implements multiple CNN architectures for illuminant estimation. We train models to classify scene illuminants into five categories (Very Warm, Warm, Neutral, Cool, Very Cool) and use Class Activation Maps (CAM) to visualize spatial attention. The project includes spatially-aware color correction using CAM-guided white balance, evaluation on LSMI (Localized Spatially Mixed Illuminant) test images, and comparison of different CAM methods (GradCAM, GradCAM++, ScoreCAM).

## Resources

### Model Weights and Training Dataset Splits
- **Google Drive**: [Download model weights and train/val/test splits](https://drive.google.com/drive/folders/1yWjb--TbCoseO4ZM1QqgypGkP8Ifm-Id?usp=sharing)
  - Contains trained model weights for all architectures
  - Includes train/val/test dataset splits

### Dataset Downloads

#### LSMI Dataset
- **Repository**: [LSMI Dataset GitHub](https://github.com/DY112/LSMI-dataset)
  - Official repository for the Large Scale Multi-Illuminant (LSMI) dataset

#### INTEL-TAU Dataset
- **Download**: [INTEL-TAU Dataset](https://etsin.fairdata.fi/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c)
  - Raw illuminant data used for training

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
│   │   ├── visualize_lsmi_masks.py      # Ground truth mask visualization
│   │   └── correct_with_cam.py          # CAM-guided color correction
│   ├── data_manipulations/
│   │   ├── augment_split_data.py        # Dataset creation and splitting
│   │   ├── balance_lsmi.py              # LSMI dataset balancing
│   │   ├── generate_lsmi_mixture_maps.py # LSMI mixture map generation
│   │   └── prepare_lsmi_test_package.py # LSMI test package preparation
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
│   │   ├── masks/                       # Ground truth masks (.npy)
│   │   └── meta.json                    # Metadata for test package
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

#### Balance LSMI Dataset
```bash
# First, balance the LSMI dataset to ensure each scene has illuminants from different clusters
# This creates lsmi_balanced.csv in the current directory
python src/data_manipulations/balance_lsmi.py
```

#### Prepare LSMI Test Package
```bash
# Then, prepare the test package using the balanced CSV
# This will generate mixture maps and copy necessary files
python src/data_manipulations/prepare_lsmi_test_package.py \
    --csv lsmi_balanced.csv \
    --src_root Data/LSMI/nikon \
    --output_dir LSMI_Test_Package \
    --meta Data/LSMI/nikon/meta.json
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
    --gt_threshold 0.0 \
    --pred_threshold 0.1 \
    --output results.csv

# Comprehensive matrix evaluation (all combinations)
python src/evaluate_masks.py \
    --matrix \
    --matrix_gt_threshold 0.0 \
    --matrix_pred_threshold 0.1 \
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

# Interactive mode
python src/display_scripts/visualize_cam.py \
    --model standard \
    --cam gradcam \
    --layer conv5 \
    --image Data/LSMI_Test_Package/images/Place101.nef \
    --interactive
```

#### Visualize Ground Truth Masks
```bash
python src/display_scripts/visualize_lsmi_masks.py \
    --image Data/LSMI_Test_Package/images/Place101.nef

# With custom package path
python src/display_scripts/visualize_lsmi_masks.py \
    --image Data/LSMI_Test_Package/images/Place101.nef \
    --package_path Data/LSMI_Test_Package
```

#### CAM-Guided Color Correction
```bash
# Single image correction
python src/display_scripts/correct_with_cam.py \
    --image Data/LSMI_Test_Package/images/Place101.nef \
    --model standard \
    --cam gradcam \
    --layer conv5

# Process multiple random images
python src/display_scripts/correct_with_cam.py \
    --model standard \
    --cam gradcam \
    --layer conv5 \
    --num-images 4

# With custom parameters
python src/display_scripts/correct_with_cam.py \
    --image Data/LSMI_Test_Package/images/Place101.nef \
    --model standard \
    --cam gradcam \
    --layer conv5 \
    --threshold 0.10 \
    --smooth_ksize 41 \
    --temp 0.7 \
    --output visualizations/cam_correction
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

This project is for educational purposes as part of CS7180 Advanced Perception course.
