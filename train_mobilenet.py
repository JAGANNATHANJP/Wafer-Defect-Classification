import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 7

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("wafer_dataset.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class_id"]
)

print("Training Samples :", len(train_df))
print("Validation Samples :", len(val_df))

# -----------------------------
# Image Loader
# -----------------------------
def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label

# -----------------------------
# TensorFlow Dataset
# -----------------------------
train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["class_id"].values
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["class_id"].values
    )
)

train_ds = train_ds.map(load_image)

val_ds = val_ds.map(load_image)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)

val_ds = val_ds.batch(BATCH_SIZE)

# -----------------------------
# MobileNetV2 Base Model
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# Build Model
# -----------------------------
model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Save Model
# -----------------------------
model.save("mobilenet_wafer.keras")

print("\nTraining Completed Successfully!")
print("Model saved as mobilenet_wafer.keras")
