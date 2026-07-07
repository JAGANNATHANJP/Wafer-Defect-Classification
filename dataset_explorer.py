import os
from collections import Counter

label_dir = "dataset/labels/labels/train"

counter = Counter()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            line = f.readline().strip()

            if line:
                class_id = int(line.split()[0])
                counter[class_id] += 1

print("Class Distribution:")
for k, v in sorted(counter.items()):
    print(f"Class {k}: {v}")
