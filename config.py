"""
config.py
======================================================
Central configuration module for the Wafer Defect
Classification Streamlit application.

This module centralizes every constant, path, and
hyperparameter used across the application so that
the rest of the codebase never hardcodes "magic values".

Author: AI Engineering Team
======================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------------------------------
# BASE PROJECT PATHS
# ------------------------------------------------------

#: Absolute path to the root directory of the project.
BASE_DIR: Path = Path(__file__).resolve().parent

#: Directory containing static assets (logos, icons, images used in UI).
ASSETS_DIR: Path = BASE_DIR / "assets"

#: Directory where generated outputs (Grad-CAM images, reports, history) are stored.
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

#: Directory containing the dataset.
DATASET_DIR: Path = BASE_DIR / "dataset" / "data"

#: Directory containing wafer images (train / validation / test).
IMAGES_DIR: Path = DATASET_DIR

#: Sub-directories for each dataset split.
TRAIN_IMAGES_DIR: Path = IMAGES_DIR / "train"
VALID_IMAGES_DIR: Path = IMAGES_DIR / "validation"
TEST_IMAGES_DIR: Path = IMAGES_DIR / "test"

#: Path to the dataset CSV file (image_path, class_id).
DATASET_CSV_PATH: Path = BASE_DIR / "wafer_dataset.csv"

#: Path to the trained Keras model file.
MODEL_PATH: Path = BASE_DIR / "best_resnet50.keras"

#: Sub-directory (inside outputs/) used to persist prediction history as JSON/CSV.
HISTORY_DIR: Path = OUTPUTS_DIR / "history"

#: Sub-directory (inside outputs/) used to store generated Grad-CAM images.
GRADCAM_OUTPUT_DIR: Path = OUTPUTS_DIR / "gradcam"

#: Sub-directory (inside outputs/) used to store downloadable prediction reports.
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

#: Optional directory holding pre-computed performance artifacts
#: (confusion matrix image, classification report, training curves).
PERFORMANCE_DIR: Path = OUTPUTS_DIR / "performance"

# Ensure runtime-writable directories exist. This is idempotent
# (exist_ok=True) and defensively guarded so that importing this
# module does not crash the app on read-only deployment filesystems
# (e.g. some managed container platforms) -- history/report saving
# will simply be unavailable in that case rather than the app failing
# to start.
for _directory in (OUTPUTS_DIR, HISTORY_DIR, GRADCAM_OUTPUT_DIR, REPORTS_DIR, PERFORMANCE_DIR):
    try:
        os.makedirs(_directory, exist_ok=True)
    except OSError:
        pass

# ------------------------------------------------------
# CLASS LABELS
# ------------------------------------------------------

#: Mapping of class_id -> human-readable wafer defect class name.
CLASS_NAMES: Dict[int, str] = {
    0: "Center",
    1: "Donut",
    2: "Edge-Loc",
    3: "Edge-Ring",
    4: "Loc",
    5: "Near-full",
    6: "Random",
    7: "Scratch",
    8: "none",
}

#: Ordered list of class names, indexed identically to model output logits.
CLASS_LABELS: List[str] = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

#: Total number of defect classes the model predicts.
NUM_CLASSES: int = len(CLASS_NAMES)

#: Color map used for color-coded prediction cards in the UI (hex codes).
CLASS_COLORS: Dict[str, str] = {
    "Center": "#4C9AFF",
    "Donut": "#9C6ADE",
    "Edge-Loc": "#FF8B00",
    "Edge-Ring": "#FF5630",
    "Loc": "#36B37E",
    "Near-full": "#00B8D9",
    "Random": "#6554C0",
    "Scratch": "#DE350B",
    "none": "#505F79",
}

# ------------------------------------------------------
# MODEL / TRAINING CONFIGURATION
# ------------------------------------------------------

#: Name of the ResNet50 backbone sub-layer inside the saved Keras model.
BACKBONE_LAYER_NAME: str = "resnet50"

#: Name of the last convolutional layer used for Grad-CAM visualization.
LAST_CONV_LAYER_NAME: str = "conv5_block3_out"

#: Input image size (height, width) expected by the model.
IMAGE_SIZE: Tuple[int, int] = (224, 224)

#: Number of color channels expected by the model input.
IMAGE_CHANNELS: int = 3

#: Full input shape (H, W, C) expected by the model.
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)

#: Architecture summary string shown on the "Model Information" page.
MODEL_ARCHITECTURE: str = (
    "Input -> ResNet50 Backbone -> GlobalAveragePooling2D -> "
    "Dropout -> Dense(256, ReLU) -> Dense(9, Softmax)"
)

#: Metadata describing how the model was trained (used for display only;
#: update these values to match the actual training run if known).
MODEL_METADATA: Dict[str, str] = {
    "Architecture": MODEL_ARCHITECTURE,
    "Backbone": "ResNet50 (ImageNet pretrained)",
    "Number of Classes": str(NUM_CLASSES),
    "Input Size": f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x {IMAGE_CHANNELS}",
    "Training Epochs": "5",
    "Optimizer": "Adam",
    "Loss Function": "Sparse Categorical Crossentropy",
    "Validation Accuracy": "N/A",
    "Validation Loss": "N/A",
}

# ------------------------------------------------------
# STREAMLIT / UI CONFIGURATION
# ------------------------------------------------------

#: Title shown in the browser tab and application header.
APP_TITLE: str = "Wafer Defect Classification System"

#: Short tagline shown under the main title.
APP_TAGLINE: str = "Deep Learning & Explainable AI for Semiconductor Wafer Inspection"

#: Emoji / icon used as the Streamlit page icon.
APP_ICON: str = "🔬"

#: Streamlit layout mode.
APP_LAYOUT: str = "wide"

#: Default sidebar state ("expanded" or "collapsed").
SIDEBAR_STATE: str = "expanded"

#: Sidebar navigation menu items, in display order.
NAV_ITEMS: List[str] = [
    "Home",
    "Image Upload & Prediction",
    "Grad-CAM Explainability",
    "Model Information",
    "Performance",
    "About",
]

#: Accepted image file extensions for upload widget.
ALLOWED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]

#: Maximum upload file size hint shown to the user, in megabytes.
MAX_UPLOAD_SIZE_MB: int = 10

#: Confidence threshold (0-1) below which predictions are flagged as low-confidence.
LOW_CONFIDENCE_THRESHOLD: float = 0.55

# ------------------------------------------------------
# GRAD-CAM CONFIGURATION
# ------------------------------------------------------

#: Opacity used when overlaying the Grad-CAM heatmap on the original image.
GRADCAM_ALPHA: float = 0.4

#: OpenCV colormap applied to the Grad-CAM heatmap.
GRADCAM_COLORMAP: str = "COLORMAP_JET"

# ------------------------------------------------------
# DEVELOPER / PROJECT INFORMATION
# ------------------------------------------------------

DEVELOPER_INFO: Dict[str, str] = {
    "Project Name": "Wafer Defect Classification using Deep Learning and Explainable AI",
    "Domain": "Semiconductor Manufacturing / Industrial AI",
    "Framework": "TensorFlow / Keras",
    "Interface": "Streamlit",
    "Explainability": "Grad-CAM",
}

# ------------------------------------------------------
# RANDOM SEED (for reproducibility of any runtime sampling)
# ------------------------------------------------------

RANDOM_SEED: int = 42
