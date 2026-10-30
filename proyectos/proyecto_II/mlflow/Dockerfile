# MLflow tracking server + model registry, published to Docker Hub as
# sebas10acosta/mlops-mlflow. Same server as Nivel_2/Taller_MLflow (MLflow
# 3.16.1 + Postgres driver + boto3), plus a configurable worker count.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# --frozen: install exactly what uv.lock says. --no-cache: don't keep uv's
# download cache inside the image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5000

# Everything comes from the environment (ConfigMap + Secret), so no secret is
# baked into the image.
#
# --artifacts-destination + --serve-artifacts: clients send model files to
# this server over HTTP and it writes them to MinIO; only this pod needs the
# MinIO credentials.
# --allowed-hosts / --cors-allowed-origins: MLflow 3 rejects Host headers and
# browser origins that are not listed (DNS rebinding / CSRF protection).
# --workers: the default of 4 used 2.3 GB in Taller MLflow; on a shared VM
# fewer workers are enough for one experiment pipeline and a few users.
CMD ["sh", "-c", "exec mlflow server \
     --host 0.0.0.0 --port 5000 \
     --workers \"${MLFLOW_WORKERS:-2}\" \
     --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" \
     --artifacts-destination \"s3://$MLFLOW_BUCKET\" \
     --serve-artifacts \
     --allowed-hosts \"$MLFLOW_ALLOWED_HOSTS\" \
     --cors-allowed-origins \"$MLFLOW_CORS_ALLOWED_ORIGINS\""]
