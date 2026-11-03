# One image for the trainer pods and the inference API:
#   sebas10acosta/mlops-diabetes-ml
#   trainer:  python -m trainer.train   (launched by Airflow's KubernetesPodOperator)
#   API:      uvicorn api.main:app       (added with the API block)
# Same environment for both, so a model is always served with the library
# versions it was trained with.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv \
    && useradd --uid 1000 --create-home app

WORKDIR /app

# Dependencies first (cached layer), then the project's own code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-cache --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    # mlflow-skinny imports GitPython; there is no git in the image.
    GIT_PYTHON_REFRESH=quiet

USER app
