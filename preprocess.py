import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32

# Load dataset
df = pd.read_csv("wafer_dataset.csv")

# Split into training and validation sets
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class_id"]
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# Function to load and preprocess images
def load_image(image_path, label):

    image = tf.io.read_file(image_path)

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


# Create TensorFlow datasets
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


# Apply preprocessing
train_ds = train_ds.map(load_image)
val_ds = val_ds.map(load_image)


# Batch and shuffle
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

print("Dataset pipeline created successfully!")
