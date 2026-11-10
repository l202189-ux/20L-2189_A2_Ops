import json
import os

import mlflow
import mlflow.keras
from mlflow import MlflowClient
from pathlib import Path
from tensorflow import keras


MODEL_NAME = "20L-2189_A2_Ops_VisionGuard"

checkpoints = [
    ("baseline", "a188a6515e334b29a1be3905f47b8006", "best_baseline.keras"),
    ("mobilenet", "bc59de2aa8ae438b9c102146c713e4c5", "tuned_best.keras"),
]

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

for label, run_id, filename in checkpoints:
    folder = Path("reports/evaluation") / label / run_id
    threshold = json.loads((folder / "threshold.json").read_text())["threshold"]

    # Reuse a version from this training run if the script is run again.
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    version = next((v for v in versions if v.run_id == run_id), None)

    if version is None:
        model = keras.models.load_model(
            Path("training_runs") / run_id / filename, compile=False
        )
        with mlflow.start_run(run_id=run_id):
            mlflow.keras.log_model(model, artifact_path="registry_model")

        version = mlflow.register_model(
            f"runs:/{run_id}/registry_model", MODEL_NAME
        )

    client.set_model_version_tag(MODEL_NAME, version.version, "architecture", label)
    client.set_model_version_tag(
        MODEL_NAME, version.version, "decision_threshold", str(threshold)
    )
    print(f"{label}: version {version.version}, threshold {threshold:.6f}")

    if label == "mobilenet":
        client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
        print(f"champion -> version {version.version}")