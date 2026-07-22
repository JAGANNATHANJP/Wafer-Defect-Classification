"""
config.py
=========================================================
Configuration for Wafer Defect Classification Training
Supports:
    • CNN V4
    • MobileNetV2 V4
    • ResNet50 V4
=========================================================
"""

from pathlib import Path
import tensorflow as tf

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "data"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"

for folder in [
    MODELS_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
]:
    folder.mkdir(exist_ok=True)

# =====================================================
# DATASET
# =====================================================

IMAGE_SIZE = (224, 224)

INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 32

SHUFFLE_BUFFER = 1000

CACHE_DATASET = True

PREFETCH_DATASET = True

NUM_CLASSES = 9

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]

AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# TRAINING
# =====================================================

EPOCHS_STAGE1 = 10

EPOCHS_STAGE2 = 40

INITIAL_LEARNING_RATE = 1e-3

FINE_TUNE_LEARNING_RATE = 1e-5

RANDOM_SEED = 42

LAYERS_TO_UNFREEZE = 50

# =====================================================
# CALLBACKS
# =====================================================

EARLY_STOPPING_PATIENCE = 8

REDUCE_LR_PATIENCE = 3

REDUCE_LR_FACTOR = 0.2

MIN_LEARNING_RATE = 1e-7

CHECKPOINT_MONITOR = "val_accuracy"

CHECKPOINT_MODE = "max"

# =====================================================
# MODEL SAVE PATHS
# =====================================================

BEST_CNN_MODEL = MODELS_DIR / "best_cnn.keras"
FINAL_CNN_MODEL = MODELS_DIR / "final_cnn.keras"

BEST_MOBILENET_MODEL = MODELS_DIR / "best_mobilenet.keras"
FINAL_MOBILENET_MODEL = MODELS_DIR / "final_mobilenet.keras"

BEST_RESNET50_MODEL = MODELS_DIR / "best_resnet50.keras"
FINAL_RESNET50_MODEL = MODELS_DIR / "final_resnet50.keras"

# =====================================================
# GPU
# =====================================================

USE_MIXED_PRECISION = True

ENABLE_XLA = True

# =====================================================
# DATA AUGMENTATION
# =====================================================

ROTATION = 0.15

ZOOM = 0.15

CONTRAST = 0.10

BRIGHTNESS = 0.10

# =====================================================
# TENSORBOARD
# =====================================================

TENSORBOARD_LOGDIR = LOGS_DIR / "tensorboard"

# =====================================================
# CSV LOGGING
# =====================================================

CSV_LOG = LOGS_DIR / "training_log.csv"