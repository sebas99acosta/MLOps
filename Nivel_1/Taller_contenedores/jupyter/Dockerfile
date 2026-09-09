FROM python:3.11-slim

# uv gestiona las dependencias del entorno (JupyterLab incluido), en vez de pip.
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.notebook_dir=/app/notebooks"]
