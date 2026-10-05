FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# --frozen: install exactly what uv.lock says, and fail if the lock is out of
# date instead of silently resolving new versions.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5000

# The backend store URI and the MinIO credentials come from the environment
# (docker-compose.yml), so no secret is baked into the image.
#
# --artifacts-destination + --serve-artifacts: clients send model files to
# this server over HTTP and the server writes them to MinIO. Only this
# container needs the MinIO credentials.
#
# --allowed-hosts: MLflow 3 rejects any Host header not on this list (DNS
# rebinding protection). Its default (localhost + private IPs) does not
# include the compose service name that Jupyter and the API use.
CMD ["sh", "-c", "exec mlflow server \
     --host 0.0.0.0 --port 5000 \
     --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" \
     --artifacts-destination \"s3://$MLFLOW_BUCKET\" \
     --serve-artifacts \
     --allowed-hosts \"$MLFLOW_ALLOWED_HOSTS\""]
