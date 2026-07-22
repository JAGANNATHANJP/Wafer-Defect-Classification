"""
predictor.py
======================================================
Model loading, preprocessing, and inference logic for
the Wafer Defect Classification application.

This module is responsible for:
    - Loading and caching the trained ResNet50 Keras model
    - Preprocessing uploaded images into model-ready tensors
    - Running inference and returning structured results

Author: AI Engineering Team
======================================================
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model

import config
import utils

logger = utils.logger


# ------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_wafer_model(model_path: Union[str, Path] = config.MODEL_PATH) -> Optional[Model]:
    """
    Load the trained wafer defect classification model from disk.

    The result is cached by Streamlit's ``cache_resource`` so the
    (potentially expensive) model deserialization only happens once
    per app session.

    Args:
        model_path: Path to the ``.keras`` model file.

    Returns:
        The loaded Keras Model instance, or None if loading failed.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error("Model file not found at %s", model_path)
        return None

    try:
        model = load_model(str(model_path), compile=False)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load model: %s. If this is a Keras-version mismatch "
            "(e.g. the model was saved with Keras 2 but loaded under Keras 3), "
            "pin `tensorflow<2.16` as specified in requirements.txt.",
            exc,
        )
        return None


def get_model_input_size(model: Model) -> Tuple[int, int]:
    """
    Determine the (height, width) input size expected by the model,
    falling back to the value defined in ``config`` if it cannot be
    inferred from the model itself.

    Args:
        model: The loaded Keras model.

    Returns:
        A (height, width) tuple.
    """
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
            return int(input_shape[1]), int(input_shape[2])
    except Exception:  # noqa: BLE001
        pass
    return config.IMAGE_SIZE


# ------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------

def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert a PIL image into a preprocessed, batched NumPy tensor
    ready to be fed into the ResNet50-based model.

    Steps performed:
        1. Resize to the target (height, width).
        2. Convert to a float32 NumPy array.
        3. Apply ``tf.keras.applications.resnet50.preprocess_input``.
        4. Add a batch dimension.

    Args:
        image: Input PIL image (RGB).
        target_size: Desired (height, width) for the model input.

    Returns:
        A NumPy array of shape (1, H, W, 3), preprocessed for ResNet50.
    """
    resized = utils.resize_image(image, target_size)
    array = utils.pil_to_array(resized)
    preprocessed = preprocess_input(array)
    batched = np.expand_dims(preprocessed, axis=0)
    return batched


# ------------------------------------------------------
# INFERENCE
# ------------------------------------------------------

def predict(
    model: Model,
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    """
    Run a full inference pass on a single image and return a
    structured result dictionary.

    Args:
        model: The loaded Keras model.
        image: Input PIL image (RGB).
        target_size: Optional override for the model input size.
            If not provided, it is inferred from the model.

    Returns:
        A dictionary with the following keys:
            - "predicted_class" (str): Name of the predicted class.
            - "predicted_class_id" (int): Index of the predicted class.
            - "confidence" (float): Confidence of the top prediction (0-1).
            - "probabilities" (Dict[str, float]): Per-class probabilities.
            - "inference_time" (float): Inference duration in seconds.
            - "preprocessed_input" (np.ndarray): The tensor fed to the model.

    Raises:
        ValueError: If the model is None or prediction fails.
    """
    if model is None:
        raise ValueError("Cannot run prediction: model is not loaded.")

    input_size = target_size or get_model_input_size(model)

    try:
        input_tensor = preprocess_image(image, input_size)

        start_time = time.perf_counter()
        raw_output = model.predict(input_tensor, verbose=0)
        elapsed_time = time.perf_counter() - start_time

        probabilities_array = _ensure_softmax(raw_output[0])

        predicted_class_id = int(np.argmax(probabilities_array))
        predicted_class = config.CLASS_NAMES.get(predicted_class_id, "Unknown")
        confidence = float(probabilities_array[predicted_class_id])

        probabilities: Dict[str, float] = {
            config.CLASS_NAMES.get(idx, str(idx)): float(prob)
            for idx, prob in enumerate(probabilities_array)
        }

        return {
            "predicted_class": predicted_class,
            "predicted_class_id": predicted_class_id,
            "confidence": confidence,
            "probabilities": probabilities,
            "inference_time": elapsed_time,
            "preprocessed_input": input_tensor,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        raise ValueError(f"Prediction failed: {exc}") from exc


def _ensure_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Ensure the model output is a valid probability distribution.

    If the raw output does not sum to ~1.0 (e.g. the model outputs
    raw logits instead of softmax probabilities), a softmax is
    applied defensively.

    Args:
        logits: Raw model output for a single sample, shape (num_classes,).

    Returns:
        A NumPy array representing a valid probability distribution.
    """
    total = float(np.sum(logits))
    if 0.98 <= total <= 1.02 and np.all(logits >= 0):
        return logits

    exp_values = np.exp(logits - np.max(logits))
    return exp_values / np.sum(exp_values)


def get_backbone(model: Model, layer_name: str = config.BACKBONE_LAYER_NAME) -> Optional[Model]:
    """
    Retrieve the ResNet50 backbone sub-model nested inside the
    full classification model.

    Args:
        model: The loaded full Keras model.
        layer_name: Name of the backbone layer (default from config).

    Returns:
        The backbone Keras Model/Layer, or None if it could not be found.
    """
    try:
        return model.get_layer(layer_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not retrieve backbone layer '%s': %s", layer_name, exc)
        return None
