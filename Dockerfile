FROM python:3.10.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements-web.txt .
RUN python -m pip install --no-cache-dir -r requirements-web.txt
COPY . .

RUN mkdir -p /app/instance

ENV MODEL_PATH=/models/tuned_best.keras \
    MODEL_THRESHOLD=0.762530 \
    MODEL_VERSION=2 \
    DATABASE_PATH=/app/instance/history.db

EXPOSE 5000

CMD ["gunicorn", "--chdir", "scripts", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "web_app:app"]