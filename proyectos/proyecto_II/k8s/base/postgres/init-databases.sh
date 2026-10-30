#!/bin/sh
# Runs once, automatically, the first time Postgres starts on an empty volume
# (docker-entrypoint-initdb.d). One instance, one database + role per
# component, so each one only reaches its own data:
#   diabetes  raw / clean / inference schemas (the three data layers)
#   mlflow    MLflow's backend store
#   airflow   Airflow's metadata database
# Tables are created by the pipeline itself, not here.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<EOSQL
CREATE ROLE diabetes LOGIN PASSWORD '${DIABETES_DB_PASSWORD}';
CREATE DATABASE diabetes OWNER diabetes;

CREATE ROLE mlflow LOGIN PASSWORD '${MLFLOW_DB_PASSWORD}';
CREATE DATABASE mlflow OWNER mlflow;

CREATE ROLE airflow LOGIN PASSWORD '${AIRFLOW_DB_PASSWORD}';
CREATE DATABASE airflow OWNER airflow;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname diabetes <<EOSQL
CREATE SCHEMA raw AUTHORIZATION diabetes;
CREATE SCHEMA clean AUTHORIZATION diabetes;
CREATE SCHEMA inference AUTHORIZATION diabetes;
EOSQL
