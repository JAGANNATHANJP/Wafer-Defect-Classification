"""
trainer.py
=========================================================
Training Engine

Supports:
    • CNN V4
    • MobileNetV2 V4
    • ResNet50 V4

Features
--------
✓ Mixed Precision
✓ XLA
✓ EarlyStopping
✓ ReduceLROnPlateau
✓ ModelCheckpoint
✓ CSVLogger
✓ TensorBoard
✓ Stage-1 Training
✓ Stage-2 Fine Tuning

=========================================================
"""

import tensorflow as tf

from training_models import unfreeze_model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard
)

from training_config import *

# =====================================================
# Enable GPU Optimizations
# =====================================================

if ENABLE_XLA:
    tf.config.optimizer.set_jit(True)

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =====================================================
# Compile Model
# =====================================================

def compile_model(model, learning_rate):

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# =====================================================
# Callbacks
# =====================================================

def get_callbacks(best_model_path):

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        CSVLogger(
            str(CSV_LOG),
            append=True,
        ),

        TensorBoard(
            log_dir=str(TENSORBOARD_LOGDIR),
        ),
    ]

    return callbacks

# =====================================================
# Stage 1 Training
# =====================================================

def train_stage1(
    model,
    train_ds,
    val_ds,
    best_model_path,
):

    print("\n==============================")
    print("Stage 1 Training")
    print("==============================")

    model = compile_model(
        model,
        INITIAL_LEARNING_RATE,
    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE1,

        callbacks=get_callbacks(best_model_path),

    )

    return history

# =====================================================
# Stage 2 Fine Tuning
# =====================================================

def fine_tune(
    model,
    train_ds,
    val_ds,
    best_model_path,
):

    print("\n==============================")
    print("Stage 2 Fine Tuning")
    print("==============================")

    model = unfreeze_model(
    model,
    layers_to_unfreeze=LAYERS_TO_UNFREEZE,
    )

    model = compile_model(
        model,
        FINE_TUNE_LEARNING_RATE,
    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE2,

        callbacks=get_callbacks(best_model_path),

    )

    return history

# =====================================================
# Save Final Model
# =====================================================

def save_final_model(
    model,
    path,
):

    model.save(path)

    print(f"\nFinal model saved to:\n{path}")