"""
utils.py
=========================================================
Utility Functions

Provides:
    • Plot training history
    • Save training history
    • Create project folders
    • Save metrics
=========================================================
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from training_config import *

# =====================================================
# Create Output Directories
# =====================================================

def create_directories():

    folders = [
        MODELS_DIR,
        LOGS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

# =====================================================
# Save Training History
# =====================================================

def save_history(history, filename="history.csv"):

    history_df = pd.DataFrame(history.history)

    save_path = LOGS_DIR / filename

    history_df.to_csv(save_path, index=False)

    print(f"Training history saved to:\n{save_path}")

# =====================================================
# Save Metrics JSON
# =====================================================

def save_metrics(metrics, filename="metrics.json"):

    save_path = RESULTS_DIR / filename

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

# =====================================================
# Plot Accuracy
# =====================================================

def plot_accuracy(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["accuracy"], label="Training")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = PLOTS_DIR / "accuracy.png"

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved:\n{save_path}")

# =====================================================
# Plot Loss
# =====================================================

def plot_loss(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training")

    plt.plot(history.history["val_loss"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = PLOTS_DIR / "loss.png"

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved:\n{save_path}")

# =====================================================
# Plot Precision
# =====================================================

def plot_precision(history):

    if "precision" not in history.history:
        return

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["precision"], label="Training")

    plt.plot(history.history["val_precision"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Precision")

    plt.title("Training Precision")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = PLOTS_DIR / "precision.png"

    plt.savefig(save_path, dpi=300)

    plt.close()

# =====================================================
# Plot Recall
# =====================================================

def plot_recall(history):

    if "recall" not in history.history:
        return

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["recall"], label="Training")

    plt.plot(history.history["val_recall"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Recall")

    plt.title("Training Recall")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = PLOTS_DIR / "recall.png"

    plt.savefig(save_path, dpi=300)

    plt.close()

# =====================================================
# Plot All Curves
# =====================================================

def plot_history(history):

    plot_accuracy(history)

    plot_loss(history)

    plot_precision(history)

    plot_recall(history)