import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

from baseline_cnn import build_baseline
from input_pipeline import BATCH_SIZE, IMAGE_SIZE, SEED, make_dataset


LEARNING_RATE = 0.001
MAX_EPOCHS = 20


class LogEpoch(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Record the rate used during this epoch, before any later reduction.
        self.learning_rate = float(self.model.optimizer.learning_rate.numpy())

    def on_epoch_end(self, epoch, logs=None):
        metrics = {name: float(value) for name, value in logs.items()}
        metrics["learning_rate_used"] = self.learning_rate
        mlflow.log_metrics(metrics, step=epoch + 1)


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri.startswith("https://dagshub.com/"):
        raise SystemExit("Load the DagsHub settings from .env before training.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("20L-2189_A2_Ops_VisionGuard")
    tf.keras.utils.set_random_seed(SEED)

    train = make_dataset("train")
    validation = make_dataset("validation")
    model = build_baseline(learning_rate=LEARNING_RATE)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    pointer = Path("data/raw.dvc")
    data_version = next(
        line.split("md5:", 1)[1].strip()
        for line in pointer.read_text().splitlines()
        if "md5:" in line
    )

    with mlflow.start_run(run_name="baseline_cnn_lr_0.001") as run:
        # A separate folder per run keeps earlier checkpoints available.
        output = Path("training_runs") / run.info.run_id
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best_baseline.keras"

        mlflow.set_tags({
            "git_commit": commit,
            "dvc_data_version": data_version,
            "seed": str(SEED),
            "logging_method": "manual",
        })
        mlflow.log_params({
            "model": "baseline_cnn",
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "augmentation": "random_rotation_0.02",
            "dropout": 0.3,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": 5,
            "lr_reduction_patience": 2,
            "lr_reduction_factor": 0.5,
            "minimum_lr": 0.00001,
        })

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2,
                min_delta=0, min_lr=0.00001, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, min_delta=0,
                restore_best_weights=True, verbose=1,
            ),
            LogEpoch(),
        ]

        history = model.fit(
            train,
            validation_data=validation,
            epochs=MAX_EPOCHS,
            callbacks=callbacks,
            verbose=2,
        ).history

        losses = history["val_loss"]
        best_index = losses.index(min(losses))
        best_epoch = best_index + 1
        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_val_loss": losses[best_index],
            "val_accuracy_at_best_epoch": history["val_accuracy"][best_index],
            "val_auc_at_best_epoch": history["val_auc"][best_index],
            "epochs_completed": len(losses),
        })

        epochs = range(1, len(losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot(epochs, history[metric], label="Training")
            ax.plot(epochs, history[f"val_{metric}"], label="Validation")
            ax.axvline(best_epoch, color="grey", linestyle="--",
                       label=f"Best validation loss: epoch {best_epoch}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend(fontsize=8)

        fig.tight_layout()
        curves = output / "training_curves.png"
        fig.savefig(curves, dpi=180)
        plt.close(fig)

        mlflow.log_artifact(str(checkpoint))
        mlflow.log_artifact(str(curves))
        mlflow.log_artifact(str(pointer))
        mlflow.log_artifact("requirements.txt")
        mlflow.log_artifacts("data/splits", artifact_path="splits")

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Best epoch: {best_epoch} of {len(losses)}")
        print(f"Validation loss: {losses[best_index]:.4f}")
        print(f"Validation accuracy: {history['val_accuracy'][best_index]:.4f}")
        print(f"Validation ROC-AUC: {history['val_auc'][best_index]:.4f}")
        print("Saved files:", output)


if __name__ == "__main__":
    main()