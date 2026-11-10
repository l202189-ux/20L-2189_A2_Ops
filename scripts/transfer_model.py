from tensorflow import keras
from tensorflow.keras import layers

from input_pipeline import IMAGE_SIZE, SEED


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def build_transfer_model():
    shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    backbone = keras.applications.MobileNetV2(
        input_shape=shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=shape, name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="scale_pixels")(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="average_features")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_score")(x)

    model = keras.Model(inputs, outputs, name="visionguard_mobilenetv2")
    compile_model(model, learning_rate=0.001)
    return model, backbone


def enable_fine_tuning(model, backbone):
    backbone.trainable = True

    for layer in backbone.layers:
        in_last_block = layer.name.startswith("block_16_")
        is_batch_norm = isinstance(layer, layers.BatchNormalization)
        layer.trainable = in_last_block and not is_batch_norm

    # Recompile after changing which weights the optimiser can update.
    compile_model(model, learning_rate=0.00001)


def show_trainable_layers(model, backbone):
    for layer in backbone.layers:
        if layer.trainable_weights:
            print(" ", layer.name)

    if model.get_layer("ai_score").trainable_weights:
        print("  ai_score")

    count = sum(weight.shape.num_elements() for weight in model.trainable_weights
                if hasattr(weight.shape, "num_elements"))
    # Keras versions that store shapes as tuples need the same product explicitly.
    if not count:
        from math import prod
        count = sum(prod(weight.shape) for weight in model.trainable_weights)
    print(f"Trainable parameters: {count:,}")


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    model, backbone = build_transfer_model()

    print("\nStage 1 - frozen backbone")
    show_trainable_layers(model, backbone)

    enable_fine_tuning(model, backbone)
    print("\nStage 2 - fine-tune the last block")
    show_trainable_layers(model, backbone)