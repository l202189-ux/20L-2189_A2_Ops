import psycopg
from psycopg.rows import dict_row


def create_table():
    with psycopg.connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                ai_score DOUBLE PRECISION NOT NULL,
                class_score DOUBLE PRECISION NOT NULL,
                decision_threshold DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL
            )
        """)


def save_prediction(filename, prediction):
    with psycopg.connect() as db:
        db.execute("""
            INSERT INTO predictions
                (filename, label, ai_score, class_score,
                 decision_threshold, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            filename,
            prediction["label"],
            float(prediction["ai_score"]),
            float(prediction["class_score"]),
            float(prediction["threshold"]),
            str(prediction["model_version"]),
        ))


def get_history():
    with psycopg.connect(row_factory=dict_row) as db:
        return db.execute("""
            SELECT id,
                   to_char(created_at AT TIME ZONE 'Asia/Karachi',
                           'YYYY-MM-DD HH24:MI:SS') AS local_time,
                   filename, label, ai_score, class_score,
                   decision_threshold, model_version
            FROM predictions
            ORDER BY id DESC
        """).fetchall()
