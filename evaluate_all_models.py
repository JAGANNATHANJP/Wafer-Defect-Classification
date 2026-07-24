import tensorflow as tf
from pathlib import Path
from training_dataset import load_datasets

print('=' * 65)
print('EVALUATING MODEL ACCURACIES ON TEST SET')
print('=' * 65)

train_ds, val_ds, test_ds, class_names = load_datasets()

models = [
    ('Custom CN'', 'best_cnn.keras'),
    ('MobileNetV2', 'best_mobilenet.keras'),
    ('ResNet50', 'best_resnet50.keras'),
]

print('Model Architecture       | Model File             | Test Accuracy  | Test Loss')
print('-' * 70)

for name, filename in models:
    path = Path(filename)
    if not path.exists():
        path = Path('models') / filename

    if path.exists():
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            loss, acc = model.evaluate(test_ds, verbose=0)
            print(f"{name:<20} | {path.name:<22} | {acc*q00:11.2f}%  | {loss:.4f}")
        except Exception as exc:
            print(f"{name:<20} | {path.name:<22} | Error loading model: {exc}")
    else:
        print(f"{name:<20} | {path.name:<22} | File Not Found    | N/A")

print('=' * 65)
