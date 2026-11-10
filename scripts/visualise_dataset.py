import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


audit_file = Path("reports/audit/image_audit.csv")
output = Path("reports/figures")
output.mkdir(parents=True, exist_ok=True)

with audit_file.open(encoding="utf-8", newline="") as file:
    images = [row for row in csv.DictReader(file) if not row["error"]]

classes = ["AI-generated", "Real"]
by_class = {}
for label in classes:
    by_class[label] = [row for row in images if row["class"] == label]

# A fixed seed gives us the same examples when we rerun the script.
rng = random.Random(42)
fig, axes = plt.subplots(2, 5, figsize=(12, 5), layout="constrained")

for row_number, label in enumerate(classes):
    examples = rng.sample(by_class[label], 5)

    for column, example in enumerate(examples):
        ax = axes[row_number, column]

        with Image.open(example["path"]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((400, 400))
            ax.imshow(preview)

        ax.set_title(f"{example['width']} × {example['height']} px", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel(label, fontsize=11)

fig.suptitle("Sample images from each class")
fig.savefig(output / "sample_images.png", dpi=180)
plt.close(fig)

# Each class uses the same bin boundaries for a fair comparison.
measurements = [
    ("width", "Width (pixels)", 1),
    ("height", "Height (pixels)", 1),
    ("size_bytes", "File size (MB)", 1_000_000),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")

for ax, (field, title, divisor) in zip(axes, measurements):
    values = []
    for label in classes:
        values.append([int(row[field]) / divisor for row in by_class[label]])

    ax.hist(values, bins=20, label=classes, color=["#386b93", "#d98645"])
    ax.set_xlabel(title)
    ax.set_ylabel("Number of images")
    ax.legend(fontsize=8)

fig.suptitle("Image dimensions and file sizes by class")
fig.savefig(output / "image_histograms.png", dpi=180)
plt.close(fig)

print("Saved:", output / "sample_images.png")
print("Saved:", output / "image_histograms.png")

print("\nFiles in the duplicate pair:")
for row in images:
    if int(row["copies_in_dataset"]) > 1:
        print(row["class"], "-", row["path"])