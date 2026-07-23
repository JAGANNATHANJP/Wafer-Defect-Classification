"""
train_cnn_v4.py
=========================================================
Train Custom CNN V4
=========================================================
"""

from training_dataset import load_datasets
from training_models import build_cnn
from training_trainer import (
    train_stage1,
    save_final_model,
)
from training_evaluate import evaluate_model

from training_config import (
    BEST_CNN_MODEL,
    FINAL_CNN_MODEL,
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

print("\nBuilding CNN Model...")

model = build_cnn()

model.summary()

# =====================================================
# Stage 1 Training
# =====================================================

history = train_stage1(
    model,
    train_ds,
    val_ds,
    BEST_CNN_MODEL,
)

# =====================================================
# Save Final Model
# =====================================================

save_final_model(
    model,
    FINAL_CNN_MODEL,
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