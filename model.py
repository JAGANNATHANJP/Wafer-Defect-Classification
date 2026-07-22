"""
model.py
=========================================================
Model Architectures

Supports

• Custom CNN
• MobileNetV2
• ResNet50

=========================================================
"""

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from training_config import *

# =========================================================
# Common Classification Head
# =========================================================

def classifier_head(x):

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.40)(x)

    x = layers.Dense(
        512,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32"
    )(x)

    return outputs

# =========================================================
# CNN
# =========================================================

def build_cnn():

    inputs = layers.Input(shape=INPUT_SHAPE)

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32,3,activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64,3,activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128,3,activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256,3,activation="relu")(x)

    outputs = classifier_head(x)

    model = models.Model(inputs,outputs,name="CNN_V4")

    return model

# =========================================================
# MobileNetV2
# =========================================================

def build_mobilenet():

    backbone = tf.keras.applications.MobileNetV2(

        include_top=False,

        weights="imagenet",

        input_shape=INPUT_SHAPE

    )

    backbone.trainable=False

    inputs = layers.Input(shape=INPUT_SHAPE)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = backbone(x,training=False)

    outputs = classifier_head(x)

    model=models.Model(inputs,outputs,name="MobileNetV2_V4")

    return model

# =========================================================
# ResNet50
# =========================================================

def build_resnet50():

    backbone = tf.keras.applications.ResNet50(

        include_top=False,

        weights="imagenet",

        input_shape=INPUT_SHAPE

    )

    backbone.trainable=False

    inputs = layers.Input(shape=INPUT_SHAPE)

    x=tf.keras.applications.resnet50.preprocess_input(inputs)

    x=backbone(x,training=False)

    outputs=classifier_head(x)

    model=models.Model(inputs,outputs,name="ResNet50_V4")

    return model

# =========================================================
# Fine Tuning
# =========================================================

def unfreeze_model(model, layers_to_unfreeze=50):

    backbone=None

    for layer in model.layers:

        if isinstance(layer,tf.keras.Model):

            backbone=layer

            break

    if backbone is None:

        return model

    backbone.trainable=True

    for layer in backbone.layers[:-layers_to_unfreeze]:

        layer.trainable=False

    return model

# =========================================================
# Summary
# =========================================================

if __name__=="__main__":

    model=build_resnet50()

    model.summary()