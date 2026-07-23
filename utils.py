"""
utils.py
======================================================
Reusable utility functions shared across the Wafer
Defect Classification Streamlit application.

Responsibilities covered by this module:
    - Image loading / preprocessing helpers
    - Dataset CSV loading
    - Prediction history persistence (JSON)
    - Downloadable prediction report generation
    - Small formatting / display helper functions

Keeping these functions here avoids duplicating logic
across app.py, predictor.py, and gradcam.py.

Author: AI Engineering Team
======================================================
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import config

# ------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------

def load_image(image_source: Union[str, Path, io.BytesIO, Any]) -> Optional[Image.Image]:
    """
    Load an image from a file path or an in-memory buffer
    (e.g. a Streamlit ``UploadedFile`` object) and convert
    it to RGB mode.

    Args:
        image_source: A path to an image file, or a file-like
            object such as a Streamlit uploaded file / BytesIO buffer.

    Returns:
        A PIL Image in RGB mode, or None if the image could
        not be loaded.
    """
    try:
        image = Image.open(image_source)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load image: %s", exc)
        return None


def resize_image(image: Image.Image, target_size: tuple = config.IMAGE_SIZE) -> Image.Image:
    """
    Resize a PIL image to the target size expected by the model.

    Args:
        image: Input PIL image.
        target_size: Desired (height, width) tuple.

    Returns:
        Resized PIL image.
    """
    # PIL resize expects (width, height); config stores (height, width).
    resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return resized


def pil_to_array(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a float32 NumPy array with shape
    (H, W, 3), without any normalization applied.

    Args:
        image: Input PIL image.

    Returns:
        NumPy array representation of the image.
    """
    return np.array(image).astype(np.float32)


def format_bytes_as_image(image: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Serialize a PIL image into raw bytes, suitable for use
    with ``st.download_button``.

    Args:
        image: PIL image to serialize.
        fmt: Output image format (e.g. "PNG", "JPEG").

    Returns:
        Raw bytes of the encoded image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


# ------------------------------------------------------
# DATASET HELPERS
# ------------------------------------------------------

def load_dataset_csv(csv_path: Union[str, Path] = config.DATASET_CSV_PATH) -> Optional[pd.DataFrame]:
    """
    Load the wafer dataset CSV file containing image paths
    and class labels.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        A pandas DataFrame with the dataset, or None if the
        file does not exist or could not be parsed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Dataset CSV not found at %s", csv_path)
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read dataset CSV: %s", exc)
        return None


def get_dataset_summary(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Compute a small summary of the dataset for display on the
    Home page (total images, per-split counts, per-class counts).

    Args:
        df: The dataset DataFrame returned by ``load_dataset_csv``.

    Returns:
        A dictionary of summary statistics. Empty if df is None.
    """
    if df is None or df.empty:
        return {}

    summary: Dict[str, Any] = {"total_images": len(df)}

    if "class_id" in df.columns:
        class_counts = df["class_id"].value_counts().sort_index()
        summary["class_distribution"] = {
            config.CLASS_NAMES.get(int(idx), str(idx)): int(count)
            for idx, count in class_counts.items()
        }

    if "image_path" in df.columns:
        for split, keyword in (("train", "train"), ("valid", "valid|validation"), ("test", "test")):
            summary[f"{split}_count"] = int(
                df["image_path"].astype(str).str.contains(keyword, regex=True).sum()
            )

    return summary


# ------------------------------------------------------
# FORMATTING HELPERS
# ------------------------------------------------------

def format_confidence(confidence: float) -> str:
    """
    Format a confidence score (0-1) as a human-readable percentage string.

    Args:
        confidence: Confidence value between 0 and 1.

    Returns:
        Formatted percentage string, e.g. "92.35%".
    """
    return f"{confidence * 100:.2f}%"


def format_inference_time(seconds: float) -> str:
    """
    Format an inference duration into a human-readable string,
    automatically choosing between milliseconds and seconds.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string, e.g. "84.21 ms" or "1.32 s".
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def get_class_color(class_name: str) -> str:
    """
    Retrieve the display color associated with a defect class.

    Args:
        class_name: Human-readable class name.

    Returns:
        A hex color code string. Falls back to a neutral gray
        if the class name is not recognized.
    """
    return config.CLASS_COLORS.get(class_name, "#6B778C")


def get_confidence_level_label(confidence: float) -> str:
    """
    Classify a confidence score into a qualitative label used
    for color-coded UI badges.

    Args:
        confidence: Confidence value between 0 and 1.

    Returns:
        One of "High", "Medium", or "Low".
    """
    if confidence >= 0.85:
        return "High"
    if confidence >= config.LOW_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


# ------------------------------------------------------
# PREDICTION HISTORY PERSISTENCE
# ------------------------------------------------------

def _history_file_path() -> Path:
    """Return the path to the JSON file used to persist prediction history."""
    return config.HISTORY_DIR / "prediction_history.json"


def load_prediction_history() -> List[Dict[str, Any]]:
    """
    Load the persisted prediction history from disk.

    Returns:
        A list of prediction record dictionaries. Empty list if
        no history file exists yet or it could not be parsed.
    """
    history_path = _history_file_path()
    if not history_path.exists():
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            if isinstance(data, list):
                return data
            return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load prediction history: %s", exc)
        return []


def append_prediction_history(record: Dict[str, Any]) -> None:
    """
    Append a single prediction record to the persisted history file.

    Args:
        record: Dictionary describing the prediction event. Expected
            keys typically include "timestamp", "predicted_class",
            "confidence", and "inference_time".
    """
    history = load_prediction_history()
    history.append(record)

    history_path = _history_file_path()
    try:
        with open(history_path, "w", encoding="utf-8") as file_handle:
            json.dump(history, file_handle, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save prediction history: %s", exc)


def build_history_record(
    predicted_class: str,
    confidence: float,
    inference_time: float,
    probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Construct a standardized prediction history record.

    Args:
        predicted_class: The predicted defect class name.
        confidence: Confidence score of the top prediction (0-1).
        inference_time: Time taken for inference, in seconds.
        probabilities: Mapping of class name to predicted probability.

    Returns:
        A dictionary ready to be appended to the prediction history.
    """
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "inference_time_seconds": round(float(inference_time), 4),
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
    }


# ------------------------------------------------------
# PREDICTION REPORT GENERATION
# ------------------------------------------------------

def generate_prediction_report_text(
    predicted_class: str,
    confidence: float,
    inference_time: float,
    probabilities: Dict[str, float],
) -> str:
    """
    Generate a plain-text prediction report suitable for download.

    Args:
        predicted_class: The predicted defect class name.
        confidence: Confidence score of the top prediction (0-1).
        inference_time: Time taken for inference, in seconds.
        probabilities: Mapping of class name to predicted probability.

    Returns:
        A formatted multi-line string report.
    """
    lines: List[str] = [
        "=" * 55,
        "WAFER DEFECT CLASSIFICATION - PREDICTION REPORT",
        "=" * 55,
        f"Generated At     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Predicted Class  : {predicted_class}",
        f"Confidence       : {format_confidence(confidence)}",
        f"Confidence Level : {get_confidence_level_label(confidence)}",
        f"Inference Time   : {format_inference_time(inference_time)}",
        "-" * 55,
        "Class Probabilities:",
    ]

    for class_name, probability in sorted(probabilities.items(), key=lambda item: -item[1]):
        lines.append(f"  {class_name:<12}: {format_confidence(probability)}")

    lines.append("=" * 55)
    lines.append("Model: ResNet50 Transfer Learning | Explainability: Grad-CAM")
    lines.append("=" * 55)

    return "\n".join(lines)


# ------------------------------------------------------
# FILE EXISTENCE HELPERS (used for Performance page)
# ------------------------------------------------------

def file_exists(path: Union[str, Path]) -> bool:
    """
    Safely check whether a given file path exists.

    Args:
        path: Path to check.

    Returns:
        True if the file exists, False otherwise.
    """
    try:
        return Path(path).exists()
    except Exception:  # noqa: BLE001
        return False
