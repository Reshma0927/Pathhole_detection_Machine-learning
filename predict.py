import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pothole_resnet50.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)


def predict_image(image_path: str):
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Put your dataset in dataset/ and run: python train.py"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)

    probability_class_1 = float(model.predict(array, verbose=0)[0][0])
    predicted_index = 1 if probability_class_1 >= 0.5 else 0
    confidence = probability_class_1 if predicted_index == 1 else 1 - probability_class_1
    label = class_names[predicted_index]

    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Note: confidence is the model score, not a guarantee of correctness.")
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Classify a road image with ResNet-50.")
    parser.add_argument("image", help="Path to a road image, e.g. test_images/road.jpg")
    args = parser.parse_args()
    predict_image(args.image)


if __name__ == "__main__":
    main()
