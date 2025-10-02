"""
Local training script for Rooster Recognition (Python 3.10 + TF 2.20.0)

This mirrors the Colab pipeline:
- Dataset structure: ./dataset/{bantam, dual_purpose, gamefowl, other}
- Preprocessing: cv2.imread -> resize(224) -> BGR2RGB -> float32/255 -> NCHW
- Model: MobileNetV2 (imagenet, include_top=False) + GAP + Dropouts + Dense head
- Training: phase 1 (frozen base), optional fine-tune top blocks
- Saving: SavedModel dir + JSON architecture + H5 weights + class_mapping.json

Run:
  .\venv310\Scripts\Activate.ps1
  python train_local_mobilenet.py
"""

import os
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
EPOCHS_PHASE1: int = 15
EPOCHS_PHASE2: int = 10
LEARNING_RATE: float = 1e-3

DATASET_DIR: str = os.path.join('.', 'dataset')
CATEGORIES: List[str] = ['bantam', 'dual_purpose', 'gamefowl', 'other']


def load_images_and_labels(dataset_dir: str, categories: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_index, category in enumerate(categories):
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_dir):
            print(f"⚠️ Missing category folder: {category_dir}")
            continue

        files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"📁 {category}: {len(files)} images")

        for fname in files:
            path = os.path.join(category_dir, fname)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(class_index)
            except Exception as e:
                print(f"⚠️ Failed reading {path}: {e}")

    if not images:
        raise RuntimeError("No images loaded. Ensure dataset/ has category subfolders with images.")

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    print(f"✅ Loaded dataset: X={X.shape}, y={y.shape}")
    return X, y


def build_model(num_classes: int) -> tf.keras.Model:
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def compute_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"⚖️ Class weights: {class_weight}")
    return class_weight


def main() -> None:
    X, y = load_images_and_labels(DATASET_DIR, CATEGORIES)
    num_classes = len(CATEGORIES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✂️ Split: train={X_train.shape[0]}, val={X_val.shape[0]}")

    class_weight = compute_weights(y_train, num_classes)

    model = build_model(num_classes)

    print("\n🚀 Phase 1: Train with frozen base model")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1
    )

    # Optional fine-tuning: unfreeze top blocks of base model
    base_model: tf.keras.Model = model.layers[0]
    unfreeze_from = max(0, len(base_model.layers) - 30)
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n🔧 Phase 2: Fine-tune top layers")
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n📊 Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

    # Save artifacts to local_model directory
    saved_dir = 'local_model'
    os.makedirs(saved_dir, exist_ok=True)
    
    # Save in Keras 3 compatible format
    try:
        model.save(saved_dir)
        print(f"✅ Model saved in SavedModel format to: {saved_dir}/")
    except Exception as e:
        print(f"⚠️ SavedModel format failed: {e}")
        # Fallback to H5 format
        try:
            model.save(os.path.join(saved_dir, 'rooster_model.h5'))
            print(f"✅ Model saved in H5 format to: {saved_dir}/rooster_model.h5")
        except Exception as e2:
            print(f"❌ H5 format also failed: {e2}")
            raise e2

    # Save class mapping
    class_mapping = {str(i): cls for i, cls in enumerate(CATEGORIES)}
    with open(os.path.join(saved_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("✅ Saved class_mapping.json")

    print("\n🎉 Training complete. Restart your Flask app to use the new model.")


if __name__ == '__main__':
    main()


