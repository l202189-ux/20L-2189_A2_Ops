import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf


SEED = 42
EPOCHS = 5
BATCH_SIZE = 64

# A later run can use DagsHub by setting MLFLOW_TRACKING_URI.
local_uri = Path("mlruns").resolve().as_uri()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or local_uri)
mlflow.set_experiment("20L-2189_A2_Ops_FashionMNIST")

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()

# Use the same small training and validation sets in both runs.
x_train = x[:5000].astype("float32") / 255
y_train = y[:5000]
x_val = x[5000:6000].astype("float32") / 255
y_val = y[5000:6000]

for learning_rate in [0.001, 0.01]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    with mlflow.start_run(run_name=f"fashion_lr_{learning_rate}"):
        mlflow.set_tag("dataset", "Fashion-MNIST")
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_units": 64,
            "seed": SEED,
            "training_images": len(x_train),
            "validation_images": len(x_val),
        })

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        ).history

        for index in range(EPOCHS):
            metrics = {name: values[index] for name, values in history.items()}
            mlflow.log_metrics(metrics, step=index + 1)

        # Save the plot and trained model with the run that produced them.
        with TemporaryDirectory() as folder:
            plot_path = Path(folder) / "loss_curve.png"
            model_path = Path(folder) / "fashion_model.keras"

            plt.plot(range(1, EPOCHS + 1), history["loss"], label="Training")
            plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(range(1, EPOCHS + 1))
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            model.save(model_path)
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(model_path))

        print(f"Learning rate {learning_rate}: "
              f"final validation accuracy = {history['val_accuracy'][-1]:.3f}")