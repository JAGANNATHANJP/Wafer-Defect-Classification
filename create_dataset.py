import os
import pandas as pd

IMAGE_DIR = "dataset/images/images/train"
LABEL_DIR = "dataset/labels/labels/train"

data = []

for image_file in os.listdir(IMAGE_DIR):

    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    label_file = image_file.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        first_line = f.readline().strip()

        if not first_line:
            continue

        class_id = int(first_line.split()[0])

    image_path = os.path.join(IMAGE_DIR, image_file)

    data.append([image_path, class_id])

df = pd.DataFrame(data, columns=["image_path", "class_id"])

print("\nDataset Summary")
print("-" * 40)
print("Total Images:", len(df))
print("\nClass Counts:")
print(df["class_id"].value_counts().sort_index())

df.to_csv("wafer_dataset.csv", index=False)

print("\nSaved: wafer_dataset.csv")
