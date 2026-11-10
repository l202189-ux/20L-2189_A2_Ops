import csv
import hashlib
from collections import Counter
from pathlib import Path
from statistics import median

from PIL import Image


class_folders = {
    "AI-generated": Path("data/raw/AiArtData/AiArtData"),
    "Real": Path("data/raw/RealArt/RealArt"),
}

output = Path("reports/audit")
output.mkdir(parents=True, exist_ok=True)

rows = []
for label, folder in class_folders.items():
    if not folder.is_dir():
        raise FileNotFoundError(f"Class folder does not exist: {folder}")

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in {"thumbs.db", "desktop.ini", ".ds_store"}:
            continue

        row = {
            "path": path.as_posix(),
            "class": label,
            "size_bytes": path.stat().st_size,
            "width": "",
            "height": "",
            "colour_mode": "",
            "format": "",
            "sha256": "",
            "error": "",
        }

        try:
            with path.open("rb") as file:
                row["sha256"] = hashlib.sha256(file.read()).hexdigest()

            with Image.open(path) as image:
                # Reading the pixels can reveal damage that the header alone misses.
                image.load()
                row["width"], row["height"] = image.size
                row["colour_mode"] = image.mode
                row["format"] = image.format

        except (OSError, ValueError, Image.DecompressionBombError) as error:
            row["error"] = str(error)

        rows.append(row)

if not rows:
    raise ValueError("The class folders contain no files to audit.")

# Matching hashes identify identical file contents, even across the two classes.
hash_counts = Counter(row["sha256"] for row in rows if row["sha256"])
for row in rows:
    row["copies_in_dataset"] = hash_counts[row["sha256"]]

with (output / "image_audit.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = []
for label in class_folders:
    class_rows = [row for row in rows if row["class"] == label]
    readable = [row for row in class_rows if not row["error"]]

    result = {
        "class": label,
        "files_checked": len(class_rows),
        "readable_images": len(readable),
        "unreadable_files": len(class_rows) - len(readable),
    }

    for field in ["width", "height", "size_bytes"]:
        values = [row[field] for row in readable]
        result[f"{field}_min"] = min(values) if values else ""
        result[f"{field}_median"] = median(values) if values else ""
        result[f"{field}_max"] = max(values) if values else ""

    summary.append(result)

with (output / "class_summary.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

for result in summary:
    print(f"{result['class']}: {result['readable_images']} readable images, "
          f"{result['unreadable_files']} unreadable files")

duplicate_groups = sum(count > 1 for count in hash_counts.values())
extra_copies = sum(count - 1 for count in hash_counts.values() if count > 1)

print("Groups of identical files:", duplicate_groups)
print("Extra copies beyond the first:", extra_copies)
print("Audit tables saved in:", output)