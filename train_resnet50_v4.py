"""
train_resnet50_v4.py
=========================================================
Train ResNet50 V4

Features
--------
✓ Transfer Learning
✓ Two-Stage Training
✓ Automatic Evaluation
✓ Save Best Model
✓ Save Final Model
✓ Training Curves
✓ Confusion Matrix
✓ Classification Report

Author: Jagannathan JP
=========================================================
"""

import tensorflow as tf

from training_dataset import load_datasets
from training_models import build_resnet50
from training_trainer import (
    train_stage1,
    fine_tune,
    save_final_model,
)
from training_evaluate import evaluate_model
from training_utils import (
    create_directories,
    save_history,
    plot_history,
)
from training_config import *

print("=" * 60)
print("Wafer Defect Classification")
print("ResNet50 V4 Training")
print("=" * 60)

# --------------------------------------------------
# Create folders
# --------------------------------------------------

create_directories()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

train_ds, val_ds, test_ds, class_names = load_datasets()

print("\nDataset Loaded Successfully")

print(f"Classes : {class_names}")

# --------------------------------------------------
# Build Model
# --------------------------------------------------

model = build_resnet50()

model.summary()

# --------------------------------------------------
# Stage 1 Training
# --------------------------------------------------

history_stage1 = train_stage1(
    model,
    train_ds,
    val_ds,
    BEST_RESNET50_MODEL,
)

save_history(history_stage1, "stage1_history.csv")

plot_history(history_stage1)

# --------------------------------------------------
# Stage 2 Fine-Tuning
# --------------------------------------------------

history_stage2 = fine_tune(
    model,
    train_ds,
    val_ds,
    BEST_RESNET50_MODEL,
)

save_history(history_stage2, "stage2_history.csv")

plot_history(history_stage2)

# --------------------------------------------------
# Save Final Model
# --------------------------------------------------

save_final_model(
    model,
    FINAL_RESNET50_MODEL,
)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

metrics = evaluate_model(
    model,
    test_ds,
    class_names,
)

print("\nFinal Metrics")

for key, value in metrics.items():
    print(f"{key:12s}: {value:.4f}")

print("\nTraining Completed Successfully!")

print("=" * 60)