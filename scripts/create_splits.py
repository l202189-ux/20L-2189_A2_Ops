import csv
import random
from collections import Counter
from pathlib import Path


rng = random.Random(42)
audit_file = Path("reports/audit/image_audit.csv")
output = Path("data/splits")

with audit_file.open(encoding="utf-8", newline="") as file:
    rows = sorted(csv.DictReader(file), key=lambda row: row["path"])

by_class = {"AI-generated": [], "Real": []}
seen = {}

for row in rows:
    if row["error"]:
        continue

    label = row["class"]
    image_hash = row["sha256"]

    # Keep the first filename when two files have identical contents.
    if image_hash in seen:
        first = seen[image_hash]
        if first["class"] != label:
            raise ValueError(f"Conflicting labels for {row['path']} and {first['path']}")
        print("Duplicate omitted:", row["path"])
        continue

    seen[image_hash] = row
    by_class[label].append(row)

splits = {"train": [], "validation": [], "test": []}

for images in by_class.values():
    rng.shuffle(images)

    # Give validation and test equal numbers from each class.
    held_out = round(len(images) * 0.15)
    splits["validation"].extend(images[:held_out])
    splits["test"].extend(images[held_out:2 * held_out])
    splits["train"].extend(images[2 * held_out:])

# Check that every distinct image appears in exactly one split.
all_hashes = [row["sha256"] for group in splits.values() for row in group]
if len(all_hashes) != len(seen) or set(all_hashes) != set(seen):
    raise ValueError("The split lists contain a repeated or missing image.")

output.mkdir(parents=True, exist_ok=True)

for name, images in splits.items():
    rng.shuffle(images)
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "class", "sha256"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(images)

    counts = Counter(row["class"] for row in images)
    print(f"{name}: {len(images)} images | "
          f"AI-generated: {counts['AI-generated']} | Real: {counts['Real']}")

print(f"Verified: {len(all_hashes)} distinct images, zero overlap.")