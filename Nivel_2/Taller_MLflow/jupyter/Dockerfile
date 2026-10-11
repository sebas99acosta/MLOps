FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# --frozen: install exactly what uv.lock says, and fail if the lock is out of
# date instead of silently resolving new versions. --no-cache: don't keep
# uv's download cache inside the image (it would roughly double its size).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8888

# The token is read from the JUPYTER_TOKEN env var.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.root_dir=/app/notebooks"]
