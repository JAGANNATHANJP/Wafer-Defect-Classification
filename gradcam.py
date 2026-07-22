"""
gradcam.py
======================================================
Grad-CAM (Gradient-weighted Class Activation Mapping)
implementation for the Wafer Defect Classification model.

The trained model wraps a ResNet50 backbone as a nested
sub-model (accessible via ``model.get_layer("resnet50")``),
followed by GlobalAveragePooling2D -> Dropout -> Dense(256)
-> Dense(7). Because the target convolutional layer
("conv5_block3_out") lives inside the nested backbone
sub-model rather than directly on the outer model's graph,
a two-stage gradient tape is used:

    1. Run the backbone as its own model to obtain both the
       last convolutional feature map AND the backbone's
       pooled output tensor, watching the conv feature map.
    2. Manually replay the remaining outer-model layers
       (GAP -> Dropout -> Dense -> Dense) on top of the
       backbone output, inside the same gradient tape, to
       reach the final class scores.

This allows gradients of the target class score to be
backpropagated all the way to the last convolutional
feature map, even though it is nested inside a sub-model.

Author: AI Engineering Team
======================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model

import config
import utils

logger = utils.logger


# ------------------------------------------------------
# GRAD-CAM HEATMAP COMPUTATION
# ------------------------------------------------------

def _build_backbone_feature_model(
    model: Model,
    backbone_layer_name: str = config.BACKBONE_LAYER_NAME,
    last_conv_layer_name: str = config.LAST_CONV_LAYER_NAME,
) -> Tuple[Model, int]:
    """
    Build an auxiliary Keras model that exposes both the last
    convolutional feature map and the final pooled output of
    the nested ResNet50 backbone.

    Args:
        model: The full classification model.
        backbone_layer_name: Name of the nested backbone layer.
        last_conv_layer_name: Name of the last convolutional layer
            inside the backbone.

    Returns:
        A tuple of (backbone_feature_model, backbone_layer_index)
        where backbone_layer_index is the position of the backbone
        layer within ``model.layers``.

    Raises:
        ValueError: If the backbone or convolutional layer cannot be found.
    """
    try:
        backbone = model.get_layer(backbone_layer_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Backbone layer '{backbone_layer_name}' not found: {exc}") from exc

    try:
        conv_layer_output = backbone.get_layer(last_conv_layer_name).output
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Convolution layer '{last_conv_layer_name}' not found inside backbone: {exc}"
        ) from exc

    backbone_feature_model = Model(
        inputs=backbone.input,
        outputs=[conv_layer_output, backbone.output],
        name="backbone_feature_extractor",
    )

    backbone_index = next(
        (idx for idx, layer in enumerate(model.layers) if layer.name == backbone_layer_name),
        None,
    )
    if backbone_index is None:
        raise ValueError(f"Could not locate backbone layer index for '{backbone_layer_name}'.")

    return backbone_feature_model, backbone_index


def make_gradcam_heatmap(
    model: Model,
    preprocessed_input: np.ndarray,
    class_index: Optional[int] = None,
    backbone_layer_name: str = config.BACKBONE_LAYER_NAME,
    last_conv_layer_name: str = config.LAST_CONV_LAYER_NAME,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a given preprocessed input tensor.

    Args:
        model: The full, loaded classification model.
        preprocessed_input: Preprocessed input tensor of shape (1, H, W, 3),
            as produced by ``predictor.preprocess_image``.
        class_index: Index of the class to explain. If None, the
            model's top predicted class is used.
        backbone_layer_name: Name of the nested backbone layer.
        last_conv_layer_name: Name of the last convolutional layer.

    Returns:
        A 2D NumPy array (H, W) heatmap normalized to the [0, 1] range.

    Raises:
        ValueError: If Grad-CAM computation fails at any stage.
    """
    try:
        backbone_feature_model, backbone_index = _build_backbone_feature_model(
            model, backbone_layer_name, last_conv_layer_name
        )
        remaining_layers = list(model.layers[backbone_index + 1:])

        input_tensor = tf.convert_to_tensor(preprocessed_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_output, backbone_output = backbone_feature_model(input_tensor, training=False)
            tape.watch(conv_output)

            # Manually replay the remaining classification head layers
            # (GlobalAveragePooling2D -> Dropout -> Dense(256) -> Dense(7))
            x = backbone_output
            for layer in remaining_layers:
                x = layer(x, training=False)
            predictions = x

            if class_index is None:
                class_index = int(tf.argmax(predictions[0]).numpy())

            class_channel = predictions[:, class_index]

        grads = tape.gradient(class_channel, conv_output)
        if grads is None:
            raise ValueError("Gradient computation returned None; check layer connectivity.")

        # Global-average-pool the gradients over width and height to
        # obtain the importance weight of each feature map channel.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU: keep only features that positively influence the target class.
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.math.reduce_max(heatmap)
        if max_value > 0:
            heatmap = heatmap / max_value

        return heatmap.numpy()

    except Exception as exc:  # noqa: BLE001
        logger.error("Grad-CAM heatmap generation failed: %s", exc)
        raise ValueError(f"Grad-CAM heatmap generation failed: {exc}") from exc


# ------------------------------------------------------
# HEATMAP OVERLAY / VISUALIZATION
# ------------------------------------------------------

def colorize_heatmap(
    heatmap: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a raw Grad-CAM heatmap to the target size and apply a
    color map for visualization.

    Args:
        heatmap: 2D heatmap array normalized to [0, 1].
        target_size: Desired (width, height) for the output image,
            matching OpenCV's (W, H) convention.

    Returns:
        A color-mapped heatmap as an RGB NumPy array (uint8).
    """
    resized_heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * resized_heatmap)
    colormap = getattr(cv2, config.GRADCAM_COLORMAP, cv2.COLORMAP_JET)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)
    return colored_heatmap_rgb


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        original_image: The original (non-preprocessed) PIL image.
        heatmap: 2D Grad-CAM heatmap normalized to [0, 1].
        alpha: Blending weight for the heatmap (0 = only original
            image, 1 = only heatmap).

    Returns:
        A tuple of (colored_heatmap_image, overlay_image), both as
        PIL Images sized to match the original image.

    Raises:
        ValueError: If overlay generation fails.
    """
    try:
        width, height = original_image.size
        colored_heatmap_rgb = colorize_heatmap(heatmap, target_size=(width, height))

        original_array = np.array(original_image).astype(np.uint8)
        overlay_array = cv2.addWeighted(
            original_array, 1 - alpha, colored_heatmap_rgb, alpha, 0
        )

        heatmap_image = Image.fromarray(colored_heatmap_rgb)
        overlay_image = Image.fromarray(overlay_array)

        return heatmap_image, overlay_image

    except Exception as exc:  # noqa: BLE001
        logger.error("Heatmap overlay generation failed: %s", exc)
        raise ValueError(f"Heatmap overlay generation failed: {exc}") from exc


def generate_gradcam_visuals(
    model: Model,
    original_image: Image.Image,
    preprocessed_input: np.ndarray,
    class_index: Optional[int] = None,
    alpha: float = config.GRADCAM_ALPHA,
) -> dict:
    """
    High-level convenience function that computes the Grad-CAM
    heatmap and produces all three visualization images
    (original, heatmap, overlay) in one call.

    Args:
        model: The full, loaded classification model.
        original_image: The original (non-preprocessed) PIL image,
            as displayed to the user.
        preprocessed_input: Preprocessed input tensor of shape (1, H, W, 3).
        class_index: Index of the class to explain. If None, the
            model's top predicted class is used.
        alpha: Blending weight for the overlay.

    Returns:
        A dictionary with keys "original_image", "heatmap_image",
        and "overlay_image", all PIL Images of matching size.
    """
    heatmap = make_gradcam_heatmap(
        model=model,
        preprocessed_input=preprocessed_input,
        class_index=class_index,
    )
    heatmap_image, overlay_image = overlay_heatmap_on_image(
        original_image=original_image, heatmap=heatmap, alpha=alpha
    )

    return {
        "original_image": original_image,
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
        "raw_heatmap": heatmap,
    }
