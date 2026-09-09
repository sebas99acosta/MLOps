FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

COPY main.py ./

EXPOSE 8025

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8025"]
