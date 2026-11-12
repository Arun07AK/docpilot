FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only torch first: avoids pulling ~2 GB of CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY docpilot ./docpilot
COPY web ./web
COPY eval ./eval
COPY bench ./bench
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s CMD curl -fs localhost:8000/healthz || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "docpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
