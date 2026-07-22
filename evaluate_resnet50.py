import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

# Load dataset
df = pd.read_csv("wafer_dataset.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class_id"]
)

# Image preprocessing
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Validation dataset
val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["class_id"].values
    )
)

val_ds = val_ds.map(load_image).batch(BATCH_SIZE)

# Load trained model
model = tf.keras.models.load_model("resnet50_wafer.keras")

# Evaluate
loss, accuracy = model.evaluate(val_ds)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Predictions
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)

y_true = []

for _, labels in val_ds:
    y_true.extend(labels.numpy())

# Classification report
print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("resnet50_confusion_matrix.png")

print("\nConfusion matrix saved as resnet50_confusion_matrix.png")
