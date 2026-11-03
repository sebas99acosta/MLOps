"""diabetes_pipeline: one batch of the source file per run, end to end.

    create_tables -> validate_source -> load_raw_batch -> quality_checks
      -> preprocess (pod) -> assign_split -> train (pod)
      -> compare_with_champion (pod) -> promote (pod)

Light tasks (SQL, reading the file) run in Airflow. Heavy ones run as pods of
the ML image (sebas10acosta/mlops-diabetes-ml) in the `mlops` namespace:
isolated memory, their own requests/limits, and exactly the library versions
the inference API uses.

Re-runnable by design:
- raw.diabetes has encounter_id as primary key and inserts use
  ON CONFLICT DO NOTHING: loading a row twice is a no-op.
- raw.batch_log maps each DAG run to its batch: a retried run reloads the same
  batch, never the next one.
- Every write happens in one transaction: a failure leaves nothing half done.
- When the whole file is loaded, load_raw_batch skips and so does the rest.
"""

import csv
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import DAG, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException, AirflowSkipException
from kubernetes.client import models as k8s
from psycopg2.extras import execute_values

SQL_DIR = Path(__file__).parent / "sql"
DB_CONN_ID = "diabetes_db"          # AIRFLOW_CONN_DIABETES_DB (Secret airflow-connections)
S3_CONN_ID = "minio_s3"             # AIRFLOW_CONN_MINIO_S3
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "datasets")
SOURCE_KEY = os.environ.get("SOURCE_KEY", "diabetes/Diabetes.csv")
BATCH_SIZE = 15_000                 # project rule: at most 15,000 rows per run
MAX_REJECTED_RATE = 0.05            # quality gate: fail the run above 5% bad rows
ML_IMAGE = os.environ.get("ML_IMAGE", "sebas10acosta/mlops-diabetes-ml:0.1.0")
ML_NAMESPACE = "mlops"
# Manual by default (one click = one batch, good for demos). Set e.g.
# DIABETES_PIPELINE_SCHEDULE="*/10 * * * *" in the Helm values to automate.
SCHEDULE = os.environ.get("DIABETES_PIPELINE_SCHEDULE") or None

NUMERIC_TEXT_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]


def _db() -> PostgresHook:
    return PostgresHook(postgres_conn_id=DB_CONN_ID)


def _raw_source_columns(cursor) -> list[str]:
    """The CSV columns expected in raw.diabetes, in table order (the DDL is the contract)."""
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'raw' AND table_name = 'diabetes'
          AND column_name NOT IN ('batch_id', 'load_timestamp', 'source_file', 'row_hash', 'status', 'status_reason')
        ORDER BY ordinal_position""")
    return [row[0] for row in cursor.fetchall()]


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _read_source_lines() -> list[str]:
    body = S3Hook(aws_conn_id=S3_CONN_ID).read_key(SOURCE_KEY, bucket_name=SOURCE_BUCKET)
    return body.splitlines()


# Trainer pods: DB password and MLflow come from the mlops namespace.
TRAINER_SECRETS = [Secret("env", "DIABETES_DB_PASSWORD", "mlops-secrets", "DIABETES_DB_PASSWORD")]
TRAINER_ENV = {
    "DIABETES_DB_HOST": "postgres",
    "DIABETES_DB_USER": "diabetes",
    "DIABETES_DB_NAME": "diabetes",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}


def trainer_pod(task_id: str, module: str, step_input: str, memory_limit: str = "1536Mi") -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"diabetes-{task_id.replace('_', '-')}",
        namespace=ML_NAMESPACE,
        image=ML_IMAGE,
        cmds=["python", "-m", f"trainer.{module}"],
        env_vars={**TRAINER_ENV, "STEP_INPUT": step_input},
        secrets=TRAINER_SECRETS,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "768Mi"},
            limits={"cpu": "2", "memory": memory_limit},
        ),
        do_xcom_push=True,          # the step writes /airflow/xcom/return.json
        get_logs=True,              # pod logs appear in the Airflow task log
        on_finish_action="delete_pod",
        image_pull_policy="IfNotPresent",
        in_cluster=True,
        startup_timeout_seconds=600,  # first run pulls the ML image
    )


with DAG(
    dag_id="diabetes_pipeline",
    description="Incremental load (<=15k rows/run) -> raw -> clean -> train -> MLflow -> @champion",
    schedule=SCHEDULE,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,               # batches must be loaded in order, one run at a time
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["proyecto2", "diabetes", "mlops"],
) as dag:

    @task
    def create_tables() -> None:
        """Idempotent DDL for the three data layers (versioned .sql files)."""
        hook = _db()
        for script in ("01_raw.sql", "02_clean.sql", "03_inference.sql"):
            hook.run((SQL_DIR / script).read_text())

    @task
    def validate_source() -> dict:
        """The source file exists in MinIO and its header matches raw.diabetes."""
        s3 = S3Hook(aws_conn_id=S3_CONN_ID)
        if not s3.check_for_key(SOURCE_KEY, bucket_name=SOURCE_BUCKET):
            raise AirflowFailException(f"Source file s3://{SOURCE_BUCKET}/{SOURCE_KEY} not found in MinIO.")
        lines = _read_source_lines()
        header = [_normalize(c) for c in next(csv.reader([lines[0]]))]
        with _db().get_conn() as conn, conn.cursor() as cur:
            expected = _raw_source_columns(cur)
        if header != expected:
            raise AirflowFailException(f"Unexpected columns. Missing: {set(expected) - set(header)}, "
                                       f"extra: {set(header) - set(expected)}")
        return {"rows_in_file": len(lines) - 1, "columns": len(header)}

    @task
    def load_raw_batch(source: dict) -> dict:
        """Insert the next slice of <= BATCH_SIZE rows, as text, into raw.diabetes."""
        run_id = get_current_context()["run_id"]
        conn = _db().get_conn()
        try:
            with conn, conn.cursor() as cur:  # one transaction
                cur.execute("SELECT batch_id, first_row, rows_in_batch, rows_inserted "
                            "FROM raw.batch_log WHERE dag_run_id = %s", (run_id,))
                if (done := cur.fetchone()) is not None:  # retry of this run: same batch
                    batch_id, first_row, rows_in_batch, rows_inserted = done
                    return {"batch_id": batch_id, "first_row": first_row,
                            "rows_in_batch": rows_in_batch, "rows_inserted": rows_inserted, "retry": True}

                cur.execute("SELECT COALESCE(MAX(batch_id), 0), COALESCE(MAX(first_row + rows_in_batch), 0) "
                            "FROM raw.batch_log")
                last_batch, first_row = cur.fetchone()
                if first_row >= source["rows_in_file"]:
                    raise AirflowSkipException(f"All {source['rows_in_file']} rows already loaded: nothing to do.")

                columns = _raw_source_columns(cur)
                lines = _read_source_lines()[1 + first_row: 1 + first_row + BATCH_SIZE]
                batch_id = last_batch + 1
                cur.execute("INSERT INTO raw.batch_log (batch_id, dag_run_id, source_file, first_row, "
                            "rows_in_batch, rows_inserted) VALUES (%s, %s, %s, %s, %s, 0)",
                            (batch_id, run_id, SOURCE_KEY, first_row, len(lines)))
                rows = [
                    (*values, batch_id, SOURCE_KEY, hashlib.md5(line.encode()).hexdigest())
                    for line, values in zip(lines, csv.reader(lines))
                ]
                inserted = execute_values(
                    cur,
                    f"INSERT INTO raw.diabetes ({', '.join(columns)}, batch_id, source_file, row_hash) "
                    "VALUES %s ON CONFLICT (encounter_id) DO NOTHING RETURNING encounter_id",
                    rows, page_size=1000, fetch=True,
                )
                cur.execute("UPDATE raw.batch_log SET rows_inserted = %s WHERE batch_id = %s",
                            (len(inserted), batch_id))
        finally:
            conn.close()
        return {"batch_id": batch_id, "first_row": first_row, "rows_in_batch": len(lines),
                "rows_inserted": len(inserted), "retry": False}

    @task
    def quality_checks(batch: dict) -> dict:
        """Mark each row of the batch valid or rejected; fail the run if too many are bad."""
        numeric_bad = " OR ".join(f"{c} !~ '^[0-9]+$'" for c in NUMERIC_TEXT_COLUMNS)
        conditions = {
            "missing encounter or patient id": "COALESCE(encounter_id, '') = '' OR COALESCE(patient_nbr, '') = ''",
            "unknown target value": "readmitted IS NULL OR readmitted NOT IN ('NO', '>30', '<30')",
            "non-numeric count or code": numeric_bad,
            "malformed age bracket": r"age !~ '^\[[0-9]+-[0-9]+\)$'",
        }
        reason = "CASE " + " ".join(f"WHEN {cond} THEN '{text}'" for text, cond in conditions.items()) + " END"
        conn = _db().get_conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"UPDATE raw.diabetes SET status = 'rejected', status_reason = {reason} "
                            f"WHERE batch_id = %s AND status = 'loaded' AND ({' OR '.join(conditions.values())})",
                            (batch["batch_id"],))
                rejected = cur.rowcount
                cur.execute("UPDATE raw.diabetes SET status = 'valid' WHERE batch_id = %s AND status = 'loaded'",
                            (batch["batch_id"],))
                valid = cur.rowcount
                total = valid + rejected
                if total and rejected / total > MAX_REJECTED_RATE:
                    # Raising rolls the transaction back: the batch stays 'loaded'.
                    raise AirflowFailException(f"{rejected}/{total} rows rejected (> {MAX_REJECTED_RATE:.0%}).")
                cur.execute("UPDATE raw.batch_log SET rows_valid = %s, rows_rejected = %s WHERE batch_id = %s",
                            (valid, rejected, batch["batch_id"]))
        finally:
            conn.close()
        return {**batch, "rows_valid": valid, "rows_rejected": rejected}

    @task
    def assign_split() -> dict:
        """train / validation / test = 70/15/15 by hash of patient_nbr (see sql/assign_split.sql)."""
        hook = _db()
        hook.run((SQL_DIR / "assign_split.sql").read_text())
        counts = hook.get_records("SELECT split, COUNT(*) FROM clean.diabetes GROUP BY split ORDER BY split")
        return {split: n for split, n in counts}

    preprocess = trainer_pod("preprocess", "preprocess", "{}")
    train = trainer_pod(
        "train", "train",
        '{"dag_run_id": "{{ run_id }}", "batches_loaded": {{ ti.xcom_pull(task_ids="load_raw_batch")["batch_id"] }}}',
        memory_limit="2Gi",
    )
    compare = trainer_pod("compare_with_champion", "compare", "{{ ti.xcom_pull(task_ids='train') | tojson }}")
    promote = trainer_pod(
        "promote", "promote",
        '{"best": {{ ti.xcom_pull(task_ids="train") | tojson }}, '
        '"decision": {{ ti.xcom_pull(task_ids="compare_with_champion") | tojson }}, '
        '"dag_run_id": "{{ run_id }}"}',
    )

    source = validate_source()
    create_tables() >> source
    checked = quality_checks(load_raw_batch(source))
    checked >> preprocess >> assign_split() >> train >> compare >> promote
