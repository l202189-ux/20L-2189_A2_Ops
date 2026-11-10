import os
import sqlite3
from contextlib import closing
from pathlib import Path


DB_PATH = Path(os.environ["DATABASE_PATH"])


def connect():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def create_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(connect()) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                ai_score REAL NOT NULL,
                class_score REAL NOT NULL,
                decision_threshold REAL NOT NULL,
                model_version TEXT NOT NULL
            )
        """)
        db.commit()


def save_prediction(filename, prediction):
    with closing(connect()) as db:
        db.execute("""
            INSERT INTO predictions
                (filename, label, ai_score, class_score,
                 decision_threshold, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            filename,
            prediction["label"],
            prediction["ai_score"],
            prediction["class_score"],
            prediction["threshold"],
            prediction["model_version"],
        ))
        db.commit()


def get_history():
    with closing(connect()) as db:
        return db.execute("""
            SELECT id, datetime(created_at, '+5 hours') AS local_time,
                   filename, label, ai_score, class_score,
                   decision_threshold, model_version
            FROM predictions
            ORDER BY id DESC
        """).fetchall()