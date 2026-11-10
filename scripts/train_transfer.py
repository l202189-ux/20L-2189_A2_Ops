import os
import subprocess
from math import prod
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from input_pipeline import BATCH_SIZE, IMAGE_SIZE, SEED, make_dataset
from transfer_model import build_transfer_model, enable_fine_tuning


def train_stage(model, stage, train, validation, tags):
    with mlflow.start_run(run_name=f"mobilenetv2_{stage}") as run:
        output = Path("training_runs") / run.info.run_id
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"{stage}_best.keras"

        mlflow.set_tags({**tags, "stage": stage})
        mlflow.log_params({
            "backbone": "MobileNetV2",
            "pretrained_weights": "imagenet",
            "initial_learning_rate": float(model.optimizer.learning_rate.numpy()),
            "image_size": IMAGE_SIZE,
            "input_batch_size": BATCH_SIZE,
            "augmentation": "random_rotation_0.02",
            "dropout": 0.3,
            "trainable_parameters": sum(
                prod(weight.shape) for weight in model.trainable_weights
            ),
        })

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2,
                min_delta=0, min_lr=1e-7, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, min_delta=0, verbose=1,
            ),
        ]

        history = model.fit(
            train, validation_data=validation,
            epochs=12, callbacks=callbacks, verbose=2,
        ).history

        losses = history["val_loss"]
        best = losses.index(min(losses))
        scores = {
            "best_epoch": best + 1,
            "best_val_loss": losses[best],
            "val_accuracy_at_best_epoch": history["val_accuracy"][best],
            "val_auc_at_best_epoch": history["val_auc"][best],
            "epochs_completed": len(losses),
        }
        mlflow.log_metrics(scores)

        epochs = range(1, len(losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot(epochs, history[metric], label="Training")
            ax.plot(epochs, history[f"val_{metric}"], label="Validation")
            ax.axvline(best + 1, color="grey", linestyle="--",
                       label=f"Best validation loss: epoch {best + 1}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend(fontsize=8)

        fig.suptitle(f"MobileNetV2 — {stage.replace('_', ' ')}")
        fig.tight_layout()
        curves = output / f"{stage}_curves.png"
        fig.savefig(curves, dpi=180)
        plt.close(fig)

        mlflow.log_artifact(str(curves))
        mlflow.log_artifact(str(checkpoint))
        mlflow.log_artifacts("data/splits", artifact_path="splits")
        mlflow.log_artifact("data/raw.dvc")
        mlflow.log_artifact("requirements.txt")

        print(f"\nStage: {stage} | Run ID: {run.info.run_id}")
        print(f"Best epoch: {best + 1} of {len(losses)}")
        print(f"Validation loss: {losses[best]:.4f}")
        print(f"Validation accuracy: {history['val_accuracy'][best]:.4f}")
        print(f"Validation ROC-AUC: {history['val_auc'][best]:.4f}")
        print("Saved files:", output)
        return checkpoint, run.info.run_id


def main():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/"):
        raise SystemExit("Load the DagsHub settings from .env before training.")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("20L-2189_A2_Ops_VisionGuard")

    # MLflow records fit parameters and per-epoch metrics automatically.
    # Our own checkpoint callback saves the model with the lowest validation loss.
    mlflow.tensorflow.autolog(
        log_models=False, log_datasets=False, checkpoint=False
    )

    tf.keras.utils.set_random_seed(SEED)
    train = make_dataset("train")
    validation = make_dataset("validation")
    model, backbone = build_transfer_model()

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    pointer = Path("data/raw.dvc").read_text()
    data_version = next(
        line.split("md5:", 1)[1].strip()
        for line in pointer.splitlines() if "md5:" in line
    )
    tags = {
        "git_commit": commit,
        "dvc_data_version": data_version,
        "seed": str(SEED),
        "logging_method": "tensorflow_autolog",
    }

    checkpoint, frozen_run = train_stage(model, "frozen", train, validation, tags)

    # Start fine-tuning from the best frozen epoch.
    model.load_weights(str(checkpoint))
    enable_fine_tuning(model, backbone)
    tags["source_frozen_run"] = frozen_run
    train_stage(model, "fine_tuned", train, validation, tags)


if __name__ == "__main__":
    main()