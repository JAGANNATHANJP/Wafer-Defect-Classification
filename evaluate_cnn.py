import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Scratch",
    "Unknown"
]

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("wafer_dataset.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class_id"]
)

# ----------------------------
# Image Loader
# ----------------------------
def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image / 255.0

    return image, label

# ----------------------------
# Validation Dataset
# ----------------------------
val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["class_id"].values
    )
)

val_ds = val_ds.map(load_image)
val_ds = val_ds.batch(BATCH_SIZE)

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model("cnn_wafer_model.keras")

# ----------------------------
# Evaluate
# ----------------------------
loss, accuracy = model.evaluate(val_ds)

print("\nValidation Loss :", loss)
print("Validation Accuracy :", accuracy)

# ----------------------------
# Predictions
# ----------------------------
predictions = model.predict(val_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = []

for images, labels in val_ds:
    y_true.extend(labels.numpy())

# ----------------------------
# Classification Report
# ----------------------------
print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(ax=ax)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as confusion_matrix.png")
