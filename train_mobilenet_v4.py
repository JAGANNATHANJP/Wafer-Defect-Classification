"""
train_mobilenet_v4.py
=========================================================
Train MobileNetV2 V4
=========================================================
"""

from training_dataset import load_datasets
from training_models import build_mobilenet
from training_trainer import (
    train_stage1,
    fine_tune,
    save_final_model,
)
from training_evaluate import evaluate_model

from training_config import (
    MOBILENET_MODEL,
    FINAL_MOBILENET_MODEL,
)

# =====================================================
# Load Dataset
# =====================================================

train_ds, val_ds, test_ds, class_names = load_datasets()

print("\nDataset Loaded Successfully")
print(f"Classes : {len(class_names)}")

# =====================================================
# Build Model
# =====================================================

print("\nBuilding MobileNetV2 Model...")

model = build_mobilenet()

model.summary()

# =====================================================
# Stage 1 Training
# =====================================================

history_stage1 = train_stage1(
    model,
    train_ds,
    val_ds,
    MOBILENET_MODEL,
)

# =====================================================
# Stage 2 Fine Tuning
# =====================================================

history_stage2 = fine_tune(
    model,
    train_ds,
    val_ds,
    MOBILENET_MODEL,
)

# =====================================================
# Save Final Model
# =====================================================

save_final_model(
    model,
    FINAL_MOBILENET_MODEL,
)

# =====================================================
# Evaluate
# =====================================================

metrics = evaluate_model(
    model,
    test_ds,
    class_names,
)

print("\nFinal Metrics\n")

for key, value in metrics.items():
    print(f"{key:12s}: {value:.4f}")

print("\nTraining Completed Successfully!")
print("=" * 60)