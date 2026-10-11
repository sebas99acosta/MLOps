# Inference API published to Docker Hub as sebas10acosta/locust-covertype-api.
# Same code as Nivel_2/Taller_MLflow/api. The image contains no model and no
# credentials: at startup it loads models:/<MODEL_NAME>@<MODEL_ALIAS> from the
# MLflow server given in MLFLOW_TRACKING_URI.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# --frozen: install exactly what uv.lock says. --no-cache: don't keep uv's
# download cache inside the image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"

COPY main.py ./

EXPOSE 8024

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8024"]
