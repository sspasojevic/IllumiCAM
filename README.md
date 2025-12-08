```
Final_Project/
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── src/                                # Source code
│   ├── __init__.py                     # Package initialization
│   ├── models/                         # Model implementations
│   │   ├── __init__.py                 # Package initialization
│   │   ├── model.py                    # IlluminantCNN (standard model)
│   │   ├── model_confidence.py         # ConfidenceWeightedCNN (FC4-inspired)
│   │   ├── model_paper.py             # ColorConstancyCNN (AlexNet-based)
│   │   └── model_illumicam3.py         # IllumiCam3 (Global Average Pooling)
│   │
│   ├── data_loader.py                  # Data loading utilities and transforms
│   │
│   ├── train.py                        # Training script (supports standard, confidence, paper, illumicam3)
│   ├── evaluate.py                     # Evaluation script for test set
│   │
│   ├── visualize_cam.py                # CAM visualization tool (GradCAM, GradCAM++, ScoreCAM)
│   ├── visualize_image.py              # Image visualization tool (CLI)
│   │
│   └── augment_split_data.py           # Data augmentation and train/val/test splitting
│
├── illuminant_eda.ipynb                # Exploratory Data Analysis notebook
│
├── cluster_centers.npy                 # Saved cluster centers (generated from EDA)
│
├── saved_models/                       # Trained model weights (excluded from git)
│   ├── best_illuminant_cnn.pth
│   ├── best_illuminant_cnn_val_8084.pth
│   ├── best_illuminant_cnn_confidence.pth
│   ├── best_paper_model.pth
│   └── best_illumicam3.pth
│
├── Data/                               # Raw dataset (excluded from git)
│   ├── Nikon_D810/
│   │   ├── field_1_cameras/           # Field images
│   │   ├── field_3_cameras/
│   │   ├── lab_printouts/
│   │   └── lab_realscene/
│   │       ├── *.tiff                  # Image files
│   │       └── *.wp                    # White point files
│   └── info/                           # Camera characterization data
│       ├── Info/
│       ├── Nikon_D810_Info/
│       └── Canon_5DSR_Info/
│
├── dataset/                            # Generated dataset (excluded from git)
│   ├── train/
│   │   ├── Cool/
│   │   ├── Neutral/
│   │   ├── Very_Cool/
│   │   ├── Very_Warm/
│   │   └── Warm/
│   ├── val/
│   │   └── [same structure as train]
│   └── test/
│       └── [same structure as train]
│
└── visualizations/                     # Generated visualizations (excluded from git)
    ├── cams/                           # CAM visualizations
    │   ├── gradcam_*_*.png
    │   ├── gradcam++_*_*.png
    │   └── scorecam_*_*.png
    ├── evaluate/                       # Evaluation results
    │   └── confusion_matrix_*.png
    ├── training_curves_*.png
    └── ...
```