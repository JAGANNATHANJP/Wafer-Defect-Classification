import tensorflow as tf
import numpy as np

IMG_SIZE = 224

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Scratch",
    "Unknown"
]

model = tf.keras.models.load_model("resnet50_wafer.keras")

image_path = input("Enter image path: ").strip()

img = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = tf.keras.utils.img_to_array(img)

img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

class_id = np.argmax(prediction)

confidence = np.max(prediction)

print("\nPrediction")
print("----------------------------")
print("Class :", CLASS_NAMES[class_id])
print("Confidence :", round(float(confidence)*100,2),"%")
