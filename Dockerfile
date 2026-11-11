# FinGuard v2 — API image (CPU-only, no CUDA driver required)
#
# Build:  docker build -t finguard .
# Run:    docker compose up --build

FROM python:3.11-slim

# xgboost needs libgomp; torch CPU needs no special system libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer unless requirements change)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-serving.txt

# Copy source code
COPY src ./src
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "finguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
