FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El entrenamiento (train.py) corre en la máquina de desarrollo; el
# contenedor solo empaqueta el API y los modelos ya entrenados.
COPY main.py .
COPY models/ ./models/

EXPOSE 8025

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8025"]
