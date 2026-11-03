"""Settings and helpers shared by the trainer steps."""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text

from diabetes_features import FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "diabetes_readmission")
MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes_readmission")
CHAMPION_ALIAS = os.environ.get("CHAMPION_ALIAS", "champion")
XCOM_PATH = Path(os.environ.get("XCOM_PATH", "/airflow/xcom/return.json"))
SPLITS = ("train", "validation", "test")


def db_engine():
    """DIABETES_DB_URL if given (local tests), else built from the pod's variables
    (the password comes from the Kubernetes Secret)."""
    url = os.environ.get("DIABETES_DB_URL")
    if not url:
        url = URL.create(
            "postgresql+psycopg2",
            username=os.environ["DIABETES_DB_USER"],
            password=os.environ["DIABETES_DB_PASSWORD"],
            host=os.environ["DIABETES_DB_HOST"],
            port=int(os.environ.get("DIABETES_DB_PORT", "5432")),
            database=os.environ["DIABETES_DB_NAME"],
        )
    return create_engine(url, pool_pre_ping=True)


def step_input() -> dict:
    """JSON passed by the DAG from earlier steps' XComs (empty if none)."""
    return json.loads(os.environ.get("STEP_INPUT") or "{}")


def write_result(result: dict) -> None:
    """Hand the step's result to Airflow (KubernetesPodOperator do_xcom_push)."""
    logging.getLogger(__name__).info("result: %s", result)
    if XCOM_PATH.parent.exists():
        XCOM_PATH.write_text(json.dumps(result))


def load_split(engine, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target of one split, in a stable order."""
    query = text(
        f"SELECT {', '.join(FEATURES)}, {TARGET} FROM clean.diabetes "
        "WHERE split = :split ORDER BY encounter_id"
    )
    data = pd.read_sql(query, engine, params={"split": split})
    if data.empty:
        raise RuntimeError(f"No rows in split '{split}': run preprocess and assign_split first.")
    return data[FEATURES], data[TARGET]
