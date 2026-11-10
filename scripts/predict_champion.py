import os
import sys

import mlflow
import mlflow.keras
import tensorflow as tf
from mlflow import MlflowClient

from input_pipeline import load_image


MODEL_NAME = "20L-2189_A2_Ops_VisionGuard"

if len(sys.argv) != 2:
    raise SystemExit('Usage: python scripts\\predict_champion.py "image_path"')

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
version = client.get_model_version_by_alias(MODEL_NAME, "champion")
threshold = float(version.tags["decision_threshold"])

# Use the resolved version so the weights and threshold come from the same version.
model = mlflow.keras.load_model(f"models:/{MODEL_NAME}/{version.version}")
image = load_image(tf.constant(sys.argv[1]))
score = float(model.predict(image[None, ...], verbose=0)[0, 0])
label = "AI-generated" if score >= threshold else "Real"

print(f"Loaded champion: {MODEL_NAME}, version {version.version}")
print(f"Image: {sys.argv[1]}")
print(f"AI score: {score:.4f}")
print(f"Decision threshold: {threshold:.4f}")
print(f"Prediction: {label}")