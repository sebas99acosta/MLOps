# Airflow image for the project, published as sebas10acosta/mlops-airflow.
# Base: the *slim* official image (Airflow core only; the default image is
# ~3.1 GB because it bundles dozens of providers) plus only the providers the
# project uses, pinned with Airflow's official constraints file so they are
# the versions this Airflow release was tested with:
#   postgres         raw / clean tables
#   amazon           S3 API of MinIO (source file)
#   cncf-kubernetes  KubernetesPodOperator (trainer pods)
#   fab              the chart's login and its admin-user job
ARG AIRFLOW_VERSION=3.2.2
ARG PYTHON_VERSION=3.12
FROM apache/airflow:slim-${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION
ARG PYTHON_VERSION
RUN pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        apache-airflow-providers-postgres \
        apache-airflow-providers-amazon \
        apache-airflow-providers-cncf-kubernetes \
        apache-airflow-providers-fab \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# DAGs and their SQL are baked into the image: a DAG change = new image tag.
COPY dags/ /opt/airflow/dags/
