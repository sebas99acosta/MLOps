# Streamlit UI: sebas10acosta/mlops-diabetes-ui
# Its own small image: the UI only calls the API over HTTP, so it carries no
# ML libraries, MLflow client or database driver.
FROM python:3.11-slim

RUN pip install --no-cache-dir streamlit==1.50.0 requests==2.32.5 \
    && useradd --uid 1000 --create-home app

WORKDIR /app
COPY app.py .

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
