import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from database import create_table, get_history, save_prediction
from inference import ImageClassifier


ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, template_folder=str(ROOT / "templates"))

MAX_FILE_BYTES = 5 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 64 * 1024

FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}

classifier = ImageClassifier()
create_table()


@app.template_filter("display_time")
def display_time(value):
    # The history query already returns Pakistan time.
    return datetime.fromisoformat(value).strftime("%d-%m-%Y %I:%M:%S %p")


def upload_error(message, status):
    return render_template("upload.html", error=message), status


@app.errorhandler(413)
def too_large(error):
    return upload_error("This file is too large. Choose an image up to 5 MiB.", 413)


@app.errorhandler(sqlite3.Error)
def database_error(error):
    app.logger.exception("Database operation failed")
    return upload_error(
        "The prediction database is unavailable. Please try again.", 503
    )


@app.get("/")
def upload():
    return render_template("upload.html")


@app.post("/result")
def result():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return upload_error("Choose an image first.", 400)

    filename = secure_filename(uploaded.filename)
    extension = Path(filename).suffix.lower()
    if extension not in FORMATS:
        return upload_error("Choose a JPG, JPEG or PNG file.", 400)

    contents = uploaded.stream.read(MAX_FILE_BYTES + 1)
    if len(contents) > MAX_FILE_BYTES:
        return too_large(None)

    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format != FORMATS[extension]:
                return upload_error(
                    "The image format does not match its filename.", 400
                )
            image.verify()

        prediction = classifier.predict(BytesIO(contents))

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return upload_error(
            "This image could not be decoded. Try another file.", 400
        )

    save_prediction(filename, prediction)
    return render_template("result.html", filename=filename, prediction=prediction)


@app.get("/history")
def history():
    return render_template("history.html", records=get_history())


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=classifier.version)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)