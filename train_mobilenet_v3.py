"""
train_mobilenet_v3.py
=========================================================
Standalone MobileNetV2 V3 Training Script
Wafer Defect Classification
=========================================================
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 40

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

# =========================================================
# Dataset
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="int",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

class_names = train_ds.class_names

print("\nClasses Found")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.10),
    ]
)

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(
    lambda x, y: (
        preprocess(augment(x, training=True)),
        y,
    ),
    num_parallel_calls=AUTOTUNE,
)

val_ds = val_ds.map(
    lambda x, y: (
        preprocess(x),
        y,
    ),
    num_parallel_calls=AUTOTUNE,
)

test_ds = test_ds.map(
    lambda x, y: (
        preprocess(x),
        y,
    ),
    num_parallel_calls=AUTOTUNE,
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# Build MobileNetV2
# =========================================================

backbone = tf.keras.applications.MobileNetV2(

    include_top=False,

    weights="imagenet",

    input_shape=INPUT_SHAPE,

)

backbone.trainable = False

inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = backbone(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.40)(x)

x = tf.keras.layers.Dense(
    512,
    activation="relu",
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu",
)(x)

x = tf.keras.layers.Dropout(0.20)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="MobileNetV2_V3",
)

model.summary()

# =========================================================
# Stage 1 Compile
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LR_STAGE1,
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
        "best_mobilenet_v3.keras",
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
        "mobilenet_v3_training.csv",
    )

)

# =========================================================
# Stage 1 Training
# =========================================================

history_stage1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=[
        checkpoint,
        earlystop,
        reduce_lr,
        csv_logger,
    ],

)

# =========================================================
# Stage 2 Fine Tuning
# =========================================================

print("\nStarting Fine Tuning...")

backbone.trainable = True

for layer in backbone.layers[:-50]:
    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LR_STAGE2,
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],

)

history_stage2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE2,

    callbacks=[
        checkpoint,
        earlystop,
        reduce_lr,
        csv_logger,
    ],

)

# =========================================================
# Load Best Model
# =========================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_mobilenet_v3.keras",
)

model = tf.keras.models.load_model(best_model_path)

print("\nLoaded Best Model")

# =========================================================
# Evaluate
# =========================================================

print("\nEvaluating on Test Dataset...")

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
# Predictions
# =========================================================

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
    "mobilenet_v3_classification_report.txt",
)

with open(report_file, "w") as f:
    f.write(report)

print(report)
print(f"\nClassification report saved to: {report_file}")

# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

plt.imshow(cm, cmap="Blues")
plt.title("MobileNetV3 Confusion Matrix")
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

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

cm_file = os.path.join(
    PLOT_DIR,
    "mobilenet_v3_confusion_matrix.png",
)

plt.savefig(
    cm_file,
    dpi=300,
)

plt.close()

print(f"Confusion matrix saved to: {cm_file}")

# =========================================================
# Training Curves
# =========================================================

acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]

loss = history_stage1.history["loss"] + history_stage2.history["loss"]
val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)

plt.plot(loss, label="Train")
plt.plot(val_loss, label="Validation")

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

curve_file = os.path.join(
    PLOT_DIR,
    "mobilenet_v3_training_curves.png",
)

plt.savefig(
    curve_file,
    dpi=300,
)

plt.close()

print(f"Training curves saved to: {curve_file}")

# =========================================================
# Save Final Model
# =========================================================

final_model = os.path.join(
    MODEL_DIR,
    "final_mobilenet_v3.keras",
)

model.save(final_model)

print(f"\nFinal model saved to:\n{final_model}")

# =========================================================
# Final Summary
# =========================================================

print("\n" + "=" * 60)
print("MOBILENET V3 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Classes              : {len(class_names)}")
print(f"Training Batches     : {len(train_ds)}")
print(f"Validation Batches   : {len(val_ds)}")
print(f"Test Batches         : {len(test_ds)}")

print(f"\nBest Model           : {best_model_path}")
print(f"Final Model          : {final_model}")
print(f"Training Curves      : {curve_file}")
print(f"Confusion Matrix     : {cm_file}")
print(f"Classification Report: {report_file}")

print("\nDone.")
print("=" * 60)