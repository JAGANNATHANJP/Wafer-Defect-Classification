import cv2
import os

img_dir = "dataset/images/images/train"

img_name = os.listdir(img_dir)[0]

img_path = os.path.join(img_dir, img_name)

img = cv2.imread(img_path)

print("Image:", img_name)
print("Shape:", img.shape)
