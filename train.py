import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "validation"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pothole_resnet50.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 123


def main():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            "Dataset folders are missing. Create dataset/train and "
            "dataset/validation, each with pothole and normal subfolders."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print("Class mapping:", {name: index for index, name in enumerate(class_names)})
    if len(class_names) != 2:
        raise ValueError("Expected exactly two folders/classes: normal and pothole.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune
    ).prefetch(autotune)
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune
    ).prefetch(autotune)

    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    backbone = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="road_image")
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pothole_probability")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, verbose=1
        ),
    ]

    print("\nTraining ResNet-50. The first run may download pretrained ImageNet weights.")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")
    print("Training complete. Use predict.py or run the Streamlit app.")


if __name__ == "__main__":
    main()
