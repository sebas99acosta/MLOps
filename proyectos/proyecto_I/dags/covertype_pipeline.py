"""Covertype collection pipeline.

Every run (once a minute) makes exactly ONE request to the Data API and
takes the new data through all three stages:

    collect_batch  ->  process_new_rows  ->  build_training_rows
    API -> raw         raw -> processed      processed -> training

Each stage only handles rows it hasn't seen yet (matched on row_hash), so a
run costs the same whether the tables hold 5k rows or 60k. Model training
is NOT done here: it lives in Jupyter and reads training.covertype.
"""

import hashlib
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_values

DATA_API_CONN_ID = "data_api"
POSTGRES_CONN_ID = "postgres_data"
GROUP_NUMBER = 5

# Exact message the API returns once the group's batch cap is reached.
CAP_REACHED_MESSAGE = "Ya se recolectó toda la información minima necesaria"

# The API returns each row as 13 strings in this order, with no header.
COLUMNS = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
    "wilderness_area",
    "soil_type",
    "cover_type",
]
NUMERIC_COLUMNS = [c for c in COLUMNS if c not in ("wilderness_area", "soil_type")]


def row_hash(row):
    """Fingerprint of a row's 13 original values: identical rows always get
    the same hash, so it works as the primary key for dedup."""
    return hashlib.md5("|".join(row).encode("utf-8")).hexdigest()


def collect_batch(**context):
    """API -> raw.covertype, plus one raw.collection_log entry."""
    run_id = context["run_id"]

    # check_response=False: a 400 must not raise automatically, because the
    # cap-reached 400 is an expected outcome we turn into a skip.
    response = HttpHook(method="GET", http_conn_id=DATA_API_CONN_ID).run(
        "/data",
        data={"group_number": GROUP_NUMBER},
        extra_options={"check_response": False, "timeout": 60},
    )

    if response.status_code == 400 and response.json().get("detail") == CAP_REACHED_MESSAGE:
        raise AirflowSkipException("Data API cap reached for this group: nothing new to collect.")
    response.raise_for_status()  # any other error (other 400s, 5xx) fails the task

    body = response.json()
    batch_number = body["batch_number"]
    rows = body["data"]

    values = [(row_hash(row), *row, batch_number, run_id) for row in rows]
    insert_sql = f"""
        INSERT INTO raw.covertype (row_hash, {", ".join(COLUMNS)}, batch_number, run_id)
        VALUES %s
        ON CONFLICT (row_hash) DO NOTHING
        RETURNING row_hash
    """

    # One transaction: the rows and their log entry are committed together,
    # or neither is (the `with conn` block commits on success, rolls back on error).
    conn = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID).get_conn()
    try:
        with conn, conn.cursor() as cursor:
            # fetch=True collects RETURNING results across ALL pages; plain
            # cursor.rowcount would only count the last page of 100.
            inserted = execute_values(cursor, insert_sql, values, fetch=True)
            rows_new = len(inserted)

            cursor.execute(
                """
                INSERT INTO raw.collection_log (run_id, batch_number, rows_received, rows_new)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE
                SET batch_number = EXCLUDED.batch_number,
                    rows_received = EXCLUDED.rows_received,
                    rows_new = EXCLUDED.rows_new,
                    fetched_at = now()
                """,
                (run_id, batch_number, len(rows), rows_new),
            )
    finally:
        conn.close()

    print(f"batch {batch_number}: received {len(rows)} rows, {rows_new} new")


# Casting a value like "abc" to INTEGER raises an error and would fail the
# whole statement. CASE guarantees the regex check runs before the cast, so
# invalid values become NULL instead of an error; rows with any NULL are then
# filtered out. (A plain `WHERE x ~ regex AND x::int > 0` would NOT be safe:
# Postgres doesn't guarantee the order it evaluates WHERE conditions in.)
_casts = ",\n        ".join(
    f"CASE WHEN r.{c} ~ '^-?[0-9]+$' THEN r.{c}::INTEGER END AS {c}" for c in NUMERIC_COLUMNS
)
_not_null = "\n      AND ".join(f"{c} IS NOT NULL" for c in NUMERIC_COLUMNS)

PROCESS_SQL = f"""
WITH typed AS (
    SELECT
        r.row_hash,
        {_casts},
        NULLIF(TRIM(r.wilderness_area), '') AS wilderness_area,
        NULLIF(TRIM(r.soil_type), '')       AS soil_type
    FROM raw.covertype r
    WHERE NOT EXISTS (SELECT 1 FROM processed.covertype p WHERE p.row_hash = r.row_hash)
)
INSERT INTO processed.covertype (row_hash, {", ".join(COLUMNS)})
SELECT row_hash, {", ".join(COLUMNS)}
FROM typed
WHERE {_not_null}
  AND wilderness_area IS NOT NULL
  AND soil_type IS NOT NULL
  AND aspect BETWEEN 0 AND 360
  AND hillshade_9am  BETWEEN 0 AND 255
  AND hillshade_noon BETWEEN 0 AND 255
  AND hillshade_3pm  BETWEEN 0 AND 255
  AND cover_type BETWEEN 0 AND 6
ON CONFLICT (row_hash) DO NOTHING
"""


def process_new_rows():
    """raw -> processed: cast to proper types, drop invalid rows (only new rows)."""
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute(PROCESS_SQL)
            print(f"processed {cursor.rowcount} new rows")  # single statement: rowcount is exact
    finally:
        conn.close()


# Split derived from row_hash: the first 7 hex characters (28 bits) as an
# integer, mod 5 -> ~20% of rows go to test, and a row's split never changes
# as more data arrives.
TRAINING_SQL = f"""
INSERT INTO training.covertype (row_hash, {", ".join(COLUMNS)}, split)
SELECT
    p.row_hash,
    {", ".join("p." + c for c in COLUMNS)},
    CASE WHEN mod(('x' || substr(p.row_hash, 1, 7))::bit(28)::int, 5) = 0
         THEN 'test' ELSE 'train' END
FROM processed.covertype p
WHERE NOT EXISTS (SELECT 1 FROM training.covertype t WHERE t.row_hash = p.row_hash)
ON CONFLICT (row_hash) DO NOTHING
"""


def build_training_rows():
    """processed -> training: add new rows with their stable train/test split."""
    conn = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID).get_conn()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute(TRAINING_SQL)
            print(f"added {cursor.rowcount} rows to training")
    finally:
        conn.close()


with DAG(
    dag_id="covertype_pipeline",
    # Every minute: ~5 requests per 5-minute batch window, more pool coverage.
    schedule=timedelta(minutes=1),
    start_date=datetime(2026, 9, 25),
    catchup=False,                      # never backfill thousands of past runs
    max_active_runs=1,                  # runs never overlap
    default_args={
        "retries": 1,
        # Short delay: with a 1-minute schedule and max_active_runs=1, the
        # default 5-minute retry delay would stall every following run.
        "retry_delay": timedelta(seconds=15),
    },
) as dag:
    collect = PythonOperator(task_id="collect_batch", python_callable=collect_batch)
    process = PythonOperator(task_id="process_new_rows", python_callable=process_new_rows)
    training = PythonOperator(task_id="build_training_rows", python_callable=build_training_rows)

    # Default trigger rule (all_success): if collect_batch skips at the cap,
    # both downstream tasks skip too — nothing new to process.
    collect >> process >> training
