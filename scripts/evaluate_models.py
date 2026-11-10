import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from input_pipeline import LABELS, make_dataset


OUTPUT = Path("reports/evaluation")
MODELS = {
    "baseline": (
        "a188a6515e334b29a1be3905f47b8006",
        "best_baseline.keras",
    ),
    "mobilenet": (
        "bc59de2aa8ae438b9c102146c713e4c5",
        "tuned_best.keras",
    ),
}


def read_rows(split):
    with Path(f"data/splits/{split}.csv").open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def get_scores(model, split, rows, folder):
    # Keep the predictions so plots and uploads can be reproduced without
    # running inference on the test set again.
    path = folder / f"{split}_predictions.csv"
    if path.exists():
        with path.open(encoding="utf-8") as file:
            saved = list(csv.DictReader(file))
        if [r["sha256"] for r in saved] != [r["sha256"] for r in rows]:
            raise ValueError(f"Saved predictions do not match {split}.csv")
        return np.array([float(r["ai_score"]) for r in saved])

    dataset = make_dataset(split, augment=False)
    scores = model.predict(dataset, verbose=0).reshape(-1)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid predictions for {split}")

    predictions = [
        {**row, "ai_score": float(score)} for row, score in zip(rows, scores)
    ]
    temporary = path.with_suffix(".tmp")
    write_csv(temporary, predictions)
    temporary.replace(path)
    return scores.astype(float)


def choose_threshold(labels, scores):
    # Zero calls every image AI. A threshold above the maximum calls every image real.
    candidates = np.unique(np.r_[0.0, scores, np.nextafter(scores.max(), np.inf)])
    best_key = None

    for threshold in candidates:
        predicted = scores >= threshold
        fp = int(np.sum((labels == 0) & predicted))
        fn = int(np.sum((labels == 1) & ~predicted))

        # For equal cost, prefer fewer false accusations, then the cut closest to 0.5.
        key = (2 * fp + fn, fp, abs(threshold - 0.5))
        if best_key is None or key < best_key:
            best_key = key
            result = {
                "threshold": float(threshold),
                "validation_cost": 2 * fp + fn,
                "validation_false_positives": fp,
                "validation_false_negatives": fn,
            }
    return result


def save_error_grid(rows, labels, predicted, scores, folder):
    errors = np.flatnonzero(labels != predicted)
    if len(errors) == 0:
        return

    rng = np.random.default_rng(42)
    selected = rng.choice(errors, size=min(10, len(errors)), replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    names = ["Real", "AI-generated"]
    entries = []

    for ax in axes.flat:
        ax.axis("off")

    for number, (ax, index) in enumerate(zip(axes.flat, selected), start=1):
        row = rows[index]
        with Image.open(row["path"]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((350, 350))
            ax.imshow(preview)

        actual = names[labels[index]]
        guess = names[predicted[index]]
        ax.set_title(f"{number}. {actual} → {guess}\nAI score {scores[index]:.3f}",
                     fontsize=9)
        entries.append({
            "number": number, "path": row["path"], "actual": actual,
            "predicted": guess, "ai_score": float(scores[index]),
        })

    fig.tight_layout()
    fig.savefig(folder / "error_grid.png", dpi=180)
    plt.close(fig)
    write_csv(folder / "error_examples.csv", entries)


def main():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/"):
        raise SystemExit("Load the DagsHub settings from .env first.")
    mlflow.set_tracking_uri(uri)

    validation_rows = read_rows("validation")
    test_rows = read_rows("test")
    validation_labels = np.array([LABELS[r["class"]] for r in validation_rows])
    test_labels = np.array([LABELS[r["class"]] for r in test_rows])

    # Select both thresholds before evaluating either model on the test set.
    prepared = []
    for name, (run_id, filename) in MODELS.items():
        folder = OUTPUT / name / run_id
        folder.mkdir(parents=True, exist_ok=True)
        model = tf.keras.models.load_model(
            Path("training_runs") / run_id / filename, compile=False
        )
        scores = get_scores(model, "validation", validation_rows, folder)
        rule = choose_threshold(validation_labels, scores)
        (folder / "threshold.json").write_text(json.dumps(rule, indent=2))
        prepared.append((name, run_id, model, folder, rule))

    summary = []
    for name, run_id, model, folder, rule in prepared:
        scores = get_scores(model, "test", test_rows, folder)
        predicted = (scores >= rule["threshold"]).astype(int)
        matrix = confusion_matrix(test_labels, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        metrics = {
            "test_accuracy": accuracy_score(test_labels, predicted),
            "test_precision": precision_score(test_labels, predicted, zero_division=0),
            "test_recall": recall_score(test_labels, predicted, zero_division=0),
            "test_f1": f1_score(test_labels, predicted, zero_division=0),
            "test_roc_auc": roc_auc_score(test_labels, scores),
        }
        result = {
            "model": name, "threshold": rule["threshold"], **metrics,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        summary.append(result)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["Real", "AI-generated"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(f"{name} — test confusion matrix")
        fig.tight_layout()
        fig.savefig(folder / "confusion_matrix.png", dpi=160)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(test_labels, scores)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"{name} — test ROC curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "roc_curve.png", dpi=160)
        plt.close(fig)

        if name == "mobilenet":
            save_error_grid(test_rows, test_labels, predicted, scores, folder)

        write_csv(folder / "test_metrics.csv", [result])
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_param("decision_threshold", rule["threshold"])
            mlflow.log_artifacts(str(folder), artifact_path="evaluation")
            mlflow.log_artifacts("data/splits", artifact_path="splits")
            mlflow.log_artifact("requirements.txt")

        print(f"\n{name}: threshold {rule['threshold']:.4f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    write_csv(OUTPUT / "test_comparison.csv", summary)
    print("\nSaved:", OUTPUT / "test_comparison.csv")


if __name__ == "__main__":
    main()