import csv
import os
import subprocess
from pathlib import Path

import keras_tuner as kt
import mlflow
import numpy as np
import tensorflow as tf

from input_pipeline import BATCH_SIZE, IMAGE_SIZE, LABELS, SEED, load_image
from train_baseline import LogEpoch
from transfer_model import build_transfer_model, compile_model


FROZEN_RUN = "0eaa27ac6a9745e48c67a705e570df9c"
FROZEN_MODEL = Path("training_runs") / FROZEN_RUN / "frozen_best.keras"
SEARCH_DIR = Path("training_runs/tuner")
REPORT_DIR = Path("reports/tuning")


def read_split(split):
    with Path(f"data/splits/{split}.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    images = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty(len(rows), dtype=np.int32)

    for index, row in enumerate(rows):
        images[index] = load_image(tf.constant(row["path"]))
        labels[index] = LABELS[row["class"]]

    print(f"Loaded {len(rows)} {split} images")
    return tf.data.Dataset.from_tensor_slices((images, labels))


def training_batches(dataset):
    dataset = dataset.shuffle(682, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    rotation = tf.keras.layers.RandomRotation(0.02, seed=SEED)

    def augment(images, labels):
        images = tf.cast(images, tf.float32)
        return rotation(images, training=True), labels

    return dataset.map(augment).prefetch(1)


class TrackedGrid(kt.GridSearch):
    def run_trial(self, trial, train_data, validation_data, tags):
        hp = trial.hyperparameters
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model, backbone = build_transfer_model()
        model.load_weights(str(FROZEN_MODEL))
        model.get_layer("dropout").rate = hp.get("dropout")

        # These are the three convolution layers verified in the layer check.
        last_block = ["block_16_expand", "block_16_depthwise", "block_16_project"]
        selected = last_block[-hp.get("unfrozen_layers"):]
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = layer.name in selected

        compile_model(model, learning_rate=hp.get("learning_rate"))

        with mlflow.start_run(run_name=f"tune_mobile_{trial.trial_id}") as run:
            mlflow.set_tags(tags)
            mlflow.log_params({
                **hp.values,
                "epochs": 2,
                "batch_size": BATCH_SIZE,
                "image_size": IMAGE_SIZE,
                "unfrozen_layer_names": ",".join(selected),
            })

            output = Path("training_runs") / run.info.run_id
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "tuned_best.keras"

            history = model.fit(
                training_batches(train_data),
                validation_data=validation_data,
                epochs=2,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        str(checkpoint), monitor="val_loss", save_best_only=True
                    ),
                    LogEpoch(),
                ],
                verbose=2,
            ).history

            best = int(np.argmin(history["val_loss"]))
            metrics = {
                "best_val_loss": history["val_loss"][best],
                "val_accuracy_at_best_epoch": history["val_accuracy"][best],
                "val_auc_at_best_epoch": history["val_auc"][best],
                "best_epoch": best + 1,
            }
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(checkpoint))

            # Keep one result per trial so the table survives an interrupted search.
            row = {
                "trial": trial.trial_id,
                **hp.values,
                **metrics,
                "run_id": run.info.run_id,
                "checkpoint": checkpoint.as_posix(),
            }
            with (SEARCH_DIR / f"result_{trial.trial_id}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

            return metrics["best_val_loss"]


def main():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/"):
        raise SystemExit("Load the DagsHub settings from .env first.")
    if not FROZEN_MODEL.is_file():
        raise SystemExit(f"Missing frozen checkpoint: {FROZEN_MODEL}")

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("20L-2189_A2_Ops_VisionGuard")

    train = read_split("train")
    validation = read_split("validation").batch(BATCH_SIZE).prefetch(1)

    hp = kt.HyperParameters()
    hp.Choice("learning_rate", [0.00001, 0.00005])
    hp.Choice("dropout", [0.2, 0.4])
    hp.Choice("unfrozen_layers", [1, 3])

    tuner = TrackedGrid(
        hypermodel=None,
        hyperparameters=hp,
        objective=kt.Objective("best_val_loss", direction="min"),
        max_trials=8,
        seed=SEED,
        directory=str(SEARCH_DIR),
        project_name="mobilenet_grid",
        overwrite=False,
    )

    pointer = Path("data/raw.dvc").read_text()
    tags = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dvc_data_version": next(
            line.split("md5:", 1)[1].strip()
            for line in pointer.splitlines() if "md5:" in line
        ),
        "seed": str(SEED),
        "source_frozen_run": FROZEN_RUN,
        "search": "mobilenet_grid",
    }

    tuner.search(train, validation, tags)

    rows = []
    for path in sorted(SEARCH_DIR.glob("result_*.csv")):
        with path.open(encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    if len(rows) != 8:
        raise SystemExit(f"Completed {len(rows)} of 8 trials. Check the trial output.")

    rows.sort(key=lambda row: float(row["best_val_loss"]))
    summary = REPORT_DIR / "visionguard_trials.csv"
    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print("\nCompleted all 8 trials.")
    print("Best learning rate:", best["learning_rate"])
    print("Best dropout:", best["dropout"])
    print("Best number of unfrozen layers:", best["unfrozen_layers"])
    print("Best validation loss:", best["best_val_loss"])
    print("Winning checkpoint:", best["checkpoint"])
    print("Full results:", summary)


if __name__ == "__main__":
    main()