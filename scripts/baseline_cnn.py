from tensorflow import keras
from tensorflow.keras import layers

from input_pipeline import IMAGE_SIZE, SEED


def build_baseline(learning_rate=0.001):
    model = keras.Sequential([
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Rescaling(1 / 255, name="scale_pixels"),

        layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv1"),
        layers.BatchNormalization(name="batch_norm"),
        layers.Activation("relu", name="relu1"),
        layers.MaxPooling2D(name="pool1"),

        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(name="pool2"),

        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3"),
        layers.GlobalAveragePooling2D(name="average_features"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="ai_score"),
    ], name="visionguard_baseline")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    model = build_baseline()
    model.summary()