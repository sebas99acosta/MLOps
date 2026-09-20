FROM apache/airflow:2.9.3

# Airflow publica "constraints files" versionadas específicamente para
# evitar que pip resuelva versiones de dependencias transitivas
# incompatibles con el propio Airflow. Es el método oficial recomendado
# para agregar paquetes sobre la imagen base — por eso usamos pip aquí
# en vez de uv, aunque en el resto del proyecto preferimos uv.
ARG AIRFLOW_VERSION=2.9.3
ARG PYTHON_VERSION=3.12

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
