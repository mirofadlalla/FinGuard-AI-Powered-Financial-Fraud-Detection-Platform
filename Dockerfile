FROM python:3.11-slim

# xgboost needs this system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "finguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
