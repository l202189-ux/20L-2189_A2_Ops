import os
import sys

from tensorflow import keras

from input_pipeline import preprocess_image


class ImageClassifier:
    def __init__(self):
        self.version = os.environ["MODEL_VERSION"]
        self.threshold = float(os.environ["MODEL_THRESHOLD"])
        self.model = keras.models.load_model(
            os.environ["MODEL_PATH"], compile=False
        )

    def predict(self, source):
        image = preprocess_image(source)
        score = float(self.model(image[None, ...], training=False).numpy()[0, 0])

        is_ai = score >= self.threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "ai_score": score,
            "class_score": score if is_ai else 1 - score,
            "threshold": self.threshold,
            "model_version": self.version,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts\\inference.py "image_path"')

    classifier = ImageClassifier()
    print(f"Loaded model version {classifier.version} once")

    for path in sys.argv[1:]:
        result = classifier.predict(path)
        print(f"\nImage: {path}")
        print(f"AI score: {result['ai_score']:.4f}")
        print(f"Decision threshold: {result['threshold']:.4f}")
        print(f"Prediction: {result['label']}")
        print(f"Score for predicted class: {result['class_score']:.4f}")