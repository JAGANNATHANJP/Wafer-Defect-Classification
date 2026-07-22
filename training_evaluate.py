"""
evaluate.py
=========================================================
Model Evaluation Module

Generates:
    • Test Accuracy
    • Precision
    • Recall
    • F1 Score
    • Classification Report
    • Confusion Matrix
    • Metrics JSON

=========================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from training_config import *

# =====================================================
# Evaluate Model
# =====================================================

def evaluate_model(
    model,
    test_ds,
    class_names,
):

    print("\nEvaluating model...")

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predicted_classes = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # =================================================
    # Classification Report
    # =================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = REPORTS_DIR / f"{model.name}_classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nSaved report to:\n{report_path}")

    # =================================================
    # Confusion Matrix
    # =================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )

    plt.tight_layout()

    cm_path = PLOTS_DIR / f"{model.name}_confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved confusion matrix to:\n{cm_path}")

    # =================================================
    # Metrics JSON
    # =================================================

    metrics = {

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),

    }

    metrics_path = RESULTS_DIR / f"{model.name}_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to:\n{metrics_path}")

    return metrics