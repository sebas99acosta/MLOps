#!/usr/bin/env bash
# Creates Airflow's Secrets in namespace `airflow` from k8s/secrets.env.
# Safe to re-run: connections are updated, generated keys are kept.
#
#   bash k8s/airflow/create-secrets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source secrets.env
set +a

NS=airflow
PG=postgres.mlops.svc.cluster.local:5432
MINIO=http://minio.mlops.svc.cluster.local:9000

kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

apply_secret() {  # apply_secret <name> --from-literal=... (create or update)
  kubectl -n "$NS" create secret generic "$@" --dry-run=client -o yaml | kubectl apply -f -
}

# Airflow's metadata database: the `airflow` database of our Postgres.
apply_secret airflow-metadata \
  --from-literal=connection="postgresql+psycopg2://airflow:${AIRFLOW_DB_PASSWORD}@${PG}/airflow"

# Connections used by the DAG (AIRFLOW_CONN_<ID> environment variables).
apply_secret airflow-connections \
  --from-literal=AIRFLOW_CONN_DIABETES_DB="postgresql://diabetes:${DIABETES_DB_PASSWORD}@${PG}/diabetes" \
  --from-literal=AIRFLOW_CONN_MINIO_S3="{\"conn_type\": \"aws\", \"login\": \"${MINIO_ROOT_USER}\", \"password\": \"${MINIO_ROOT_PASSWORD}\", \"extra\": {\"endpoint_url\": \"${MINIO}\", \"region_name\": \"us-east-1\"}}"

# Keys generated once and then kept: rotating them on every run would
# invalidate sessions and make encrypted values unreadable.
for pair in airflow-fernet:fernet-key airflow-api-secret:api-secret-key airflow-jwt-secret:jwt-secret; do
  name=${pair%%:*}
  key=${pair##*:}
  if ! kubectl -n "$NS" get secret "$name" >/dev/null 2>&1; then
    value=$(python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())")
    kubectl -n "$NS" create secret generic "$name" --from-literal="$key=$value"
  fi
done

kubectl -n "$NS" get secrets
