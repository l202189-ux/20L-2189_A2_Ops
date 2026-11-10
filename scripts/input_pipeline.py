import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


IMAGE_SIZE = 160
BATCH_SIZE = 16
SEED = 42
LABELS = {"Real": 0, "AI-generated": 1}


def load_image(path):
    # Use the same decoder as the audit so the accepted formats stay consistent.
    with Image.open(path.numpy().decode("utf-8")) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.pad(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
            color="black",
        )
        return np.asarray(image, dtype=np.uint8)


def prepare_image(path, label):
    image = tf.py_function(load_image, [path], Tout=tf.uint8)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return tf.cast(image, tf.float32), label


def make_dataset(split, augment=True):
    with Path(f"data/splits/{split}.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    paths = [row["path"] for row in rows]
    labels = [LABELS[row["class"]] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split == "train":
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    # Some originals are large. Decode one at a time to limit memory use.
    dataset = dataset.map(prepare_image, num_parallel_calls=1)
    dataset = dataset.batch(BATCH_SIZE)

    if split == "train" and augment:
        rotation = tf.keras.layers.RandomRotation(0.02, seed=SEED)
        dataset = dataset.map(
            lambda images, labels: (rotation(images, training=True), labels)
        )

    return dataset.prefetch(1)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)

    for split in ["train", "validation"]:
        dataset = make_dataset(split)
        total = 0

        for batch_number, (images, labels) in enumerate(dataset):
            total += int(images.shape[0])

            if split == "train" and batch_number == 0:
                print("First training batch:", images.shape)
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))

                for ax, image, label in zip(axes.flat, images.numpy(), labels.numpy()):
                    ax.imshow(np.clip(image, 0, 255).astype("uint8"))
                    ax.set_title("AI-generated" if label == 1 else "Real")
                    ax.axis("off")

                fig.tight_layout()
                output = Path("reports/figures/augmented_training_batch.png")
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=160)
                plt.close(fig)
                print("Saved:", output)

        print(f"{split}: {total} images loaded")