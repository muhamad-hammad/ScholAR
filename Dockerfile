FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY main.py .
COPY data/ ./data/

ENV PYTHONPATH=/app \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/cache

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}
