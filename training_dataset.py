"""
dataset.py
=========================================================
Dataset Loader for Wafer Defect Classification

Supports:
    - CNN V4
    - MobileNetV2 V4
    - ResNet50 V4

Author: Jagannathan JP
=========================================================
"""

import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers

from training_config import *

# =====================================================
# GPU / Dataset Optimization
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE


def ensure_dataset_extracted():
    """Extracts data.zip into dataset/ if TRAIN_DIR does not exist."""
    if not TRAIN_DIR.exists():
        zip_path = BASE_DIR / "data.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path} into {BASE_DIR / 'dataset'}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(BASE_DIR / "dataset")
            print("Dataset extracted successfully!")
        else:
            raise FileNotFoundError(
                f"Dataset training directory not found at '{TRAIN_DIR}' and zip file '{zip_path}' is missing."
            )


# =====================================================
# Data Augmentation
# =====================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(ROTATION),
        layers.RandomZoom(ZOOM),
        layers.RandomContrast(CONTRAST),
    ],
    name="data_augmentation",
)

# =====================================================
# Dataset Loader
# =====================================================

def load_datasets():
    """
    Loads train, validation and test datasets.

    Returns
    -------
    train_ds
    val_ds
    test_ds
    class_names
    """
    ensure_dataset_extracted()

    train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    print("\nClasses Found:")
    print(class_names)

    # -------------------------------------------------
    # Apply Data Augmentation (Training Only)
    # -------------------------------------------------

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # -------------------------------------------------
    # ResNet50 / MobileNetV2 Preprocessing
    # -------------------------------------------------

    preprocess = tf.keras.applications.resnet50.preprocess_input

    train_ds = train_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    # -------------------------------------------------
    # Performance Optimization
    # -------------------------------------------------

    if CACHE_DATASET:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    train_ds = train_ds.shuffle(
        SHUFFLE_BUFFER,
        seed=RANDOM_SEED,
    )

    if PREFETCH_DATASET:
        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# =====================================================
# Dataset Information
# =====================================================

def dataset_info():

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\n===================================")
    print("Dataset Information")
    print("===================================")

    print(f"Classes          : {len(class_names)}")
    print(f"Image Size       : {IMAGE_SIZE}")
    print(f"Batch Size       : {BATCH_SIZE}")

    print("\nClass Names:")

    for i, name in enumerate(class_names):
        print(f"{i}: {name}")
    print()
    print(f"Training batches   : {len(train_ds)}")
    print(f"Validation batches : {len(val_ds)}")
    print(f"Test batches       : {len(test_ds)}")

    print("===================================")


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":
    dataset_info()