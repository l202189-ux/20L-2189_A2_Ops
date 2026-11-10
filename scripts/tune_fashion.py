import csv
from pathlib import Path

import keras_tuner as kt
import mlflow
import tensorflow as tf

from train_baseline import LogEpoch


SEED = 42


def build_model(hp):
    units = hp.Choice("units", [32, 64])
    learning_rate = hp.Choice("learning_rate", [0.001, 0.01])

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class TrackedSearch(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        tf.keras.utils.set_random_seed(SEED)
        with mlflow.start_run(run_name=f"fashion_tuner_{trial.trial_id}"):
            mlflow.log_params(trial.hyperparameters.values)
            mlflow.log_params({"epochs": 2, "batch_size": 64, "seed": SEED})
            kwargs["callbacks"] = [LogEpoch()]
            return super().run_trial(trial, *args, **kwargs)


def main():
    mlflow.set_experiment("20L-2189_A2_Ops_FashionTuner")
    (images, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = images[:5000].astype("float32") / 255
    x_val = images[5000:6000].astype("float32") / 255

    tuner = TrackedSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        seed=SEED,
        directory="training_runs/tuner",
        project_name="fashion_random",
        overwrite=False,
    )
    tuner.search(
        x_train, labels[:5000],
        validation_data=(x_val, labels[5000:6000]),
        epochs=2,
        batch_size=64,
        verbose=2,
    )

    output = Path("reports/tuning/fashion_trials.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = tuner.oracle.get_best_trials(4)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["trial", "units", "learning_rate", "best_val_accuracy"])
        for trial in trials:
            hp = trial.hyperparameters
            writer.writerow([
                trial.trial_id, hp.get("units"),
                hp.get("learning_rate"), trial.score,
            ])

    best = trials[0]
    print("\nBest hidden units:", best.hyperparameters.get("units"))
    print("Best learning rate:", best.hyperparameters.get("learning_rate"))
    print(f"Best validation accuracy: {best.score:.4f}")
    print("Saved:", output)


if __name__ == "__main__":
    main()