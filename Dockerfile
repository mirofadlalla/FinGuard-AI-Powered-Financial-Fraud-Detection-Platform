# FinGuard v2 — API serving image (CPU-only, no CUDA driver required)
#
# Build:  docker build -t finguard .
# Run:    docker compose up --build
#
# This image contains ONLY the serving stack.
# The model artifacts are mounted as a read-only volume at runtime (see docker-compose.yml).

FROM python:3.11-slim

# xgboost needs libgomp for multi-threading
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer is cached unless requirements change)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-serving.txt

# Copy only the serving packages (data, scripts, tests, docs stay out of image)
COPY api        ./api
COPY assistant  ./assistant
COPY database   ./database
COPY evaluation ./evaluation
COPY features   ./features
COPY models     ./models
COPY monitoring ./monitoring
COPY pipelines  ./pipelines

ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1000 finguard
USER finguard

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
