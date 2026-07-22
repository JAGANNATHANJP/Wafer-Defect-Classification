"""
train_cnn_v3.py
=========================================================
Standalone CNN V3 Training Script
Wafer Defect Classification
=========================================================
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# =========================================================
# Reproducibility
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# Paths
# =========================================================

TRAIN_DIR = "dataset/data/train"
VALID_DIR = "dataset/data/validation"
TEST_DIR = "dataset/data/test"

MODEL_DIR = "models"
PLOT_DIR = "plots"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =========================================================
# Parameters
# =========================================================

IMAGE_SIZE = (224, 224)

INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 32

NUM_CLASSES = 9

EPOCHS = 50

INITIAL_LR = 1e-3

# =========================================================
# Dataset
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names

print("\nClasses Found:")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ]
)

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE,
)

normalizer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalizer(x), y),
    num_parallel_calls=AUTOTUNE,
)

val_ds = val_ds.map(
    lambda x, y: (normalizer(x), y),
    num_parallel_calls=AUTOTUNE,
)

test_ds = test_ds.map(
    lambda x, y: (normalizer(x), y),
    num_parallel_calls=AUTOTUNE,
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# CNN Model
# =========================================================

model = models.Sequential(

    [

        layers.Input(shape=INPUT_SHAPE),

        layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.40),

        layers.Dense(
            512,
            activation="relu",
        ),

        layers.Dropout(0.30),

        layers.Dense(
            256,
            activation="relu",
        ),

        layers.Dropout(0.20),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax",
        ),

    ]

)

model.summary()

# =========================================================
# Compile
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],

)

# =========================================================
# Callbacks
# =========================================================

checkpoint = callbacks.ModelCheckpoint(

    filepath=os.path.join(
        MODEL_DIR,
        "best_cnn_v3.keras",
    ),

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1,

)

earlystop = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True,

    verbose=1,

)

reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=1e-7,

    verbose=1,

)

csv_logger = callbacks.CSVLogger(

    os.path.join(
        MODEL_DIR,
        "cnn_training_v3.csv",
    )

)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        earlystop,
        reduce_lr,
        csv_logger,
    ],

)

# =========================================================
# Training Curves
# =========================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plot_path = os.path.join(
    PLOT_DIR,
    "cnn_v3_training_curves.png",
)

plt.savefig(
    plot_path,
    dpi=300,
)

plt.close()

print(f"\nTraining curves saved to: {plot_path}")

# =========================================================
# Evaluation
# =========================================================

print("\nEvaluating model...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_loss, test_acc, test_precision, test_recall = model.evaluate(
    test_ds,
    verbose=1,
)

print("\n======================================")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print("======================================")

# =========================================================
# Classification Report
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
)

report_file = os.path.join(
    REPORT_DIR,
    "cnn_v3_classification_report.txt",
)

with open(report_file, "w") as f:
    f.write(report)

print(report)
print(f"\nClassification report saved to: {report_file}")

# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

plt.figure(figsize=(10, 8))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues",
)

plt.title("CNN V3 Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(class_names))

plt.xticks(
    ticks,
    class_names,
    rotation=45,
)

plt.yticks(
    ticks,
    class_names,
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

cm_path = os.path.join(
    PLOT_DIR,
    "cnn_v3_confusion_matrix.png",
)

plt.savefig(
    cm_path,
    dpi=300,
)

plt.close()

print(f"Confusion matrix saved to: {cm_path}")

# =========================================================
# Save Final Model
# =========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_cnn_v3.keras",
)

model.save(final_model_path)

print(f"\nFinal model saved to:\n{final_model_path}")

# =========================================================
# Save Training History
# =========================================================

history_file = os.path.join(
    REPORT_DIR,
    "cnn_v3_history.txt",
)

with open(history_file, "w") as f:

    for key in history.history:

        f.write(f"{key}\n")

        for value in history.history[key]:
            f.write(f"{value}\n")

        f.write("\n")

print(f"Training history saved to: {history_file}")

# =========================================================
# Final Summary
# =========================================================

print("\n" + "=" * 60)
print("CNN V3 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Classes              : {len(class_names)}")
print(f"Training Images      : {len(train_ds)} batches")
print(f"Validation Images    : {len(val_ds)} batches")
print(f"Test Images          : {len(test_ds)} batches")

print(f"\nBest Model           : {os.path.join(MODEL_DIR, 'best_cnn_v3.keras')}")
print(f"Final Model          : {final_model_path}")
print(f"Training Curves      : {plot_path}")
print(f"Confusion Matrix     : {cm_path}")
print(f"Classification Report: {report_file}")

print("\nDone.")
print("=" * 60)