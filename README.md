# Wafer Defect Classification using Deep Learning

## Overview

This project classifies semiconductor wafer defects using deep learning.

Three models were implemented and compared:

- Custom CNN
- MobileNetV2
- ResNet50 (Best Model)

---

## Dataset

- 3170 training images
- 7 defect classes

Classes:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Scratch
- Unknown

---

## Results

| Model | Validation Accuracy |
|--------|---------------------|
| CNN | 49.68% |
| MobileNetV2 | 65.62% |
| ResNet50 | **70.03%** |

---

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Run Training

```bash
python3 train_resnet50.py
