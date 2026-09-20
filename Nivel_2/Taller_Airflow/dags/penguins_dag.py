import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MYSQL_CONN_ID = "mysql_penguins"
RAW_TABLE = "penguins_raw"
PROCESSED_TABLE = "penguins_processed"
CSV_PATH = "/opt/airflow/data/penguins.csv"
MODELS_DIR = Path("/opt/airflow/models")

FEATURE_COLS = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
TARGET_COL = "species"


def delete_content():
    """Task 1: wipes the database clean before every run.

    Uses DROP TABLE IF EXISTS (not DELETE/TRUNCATE) because on the very
    first run ever, neither table exists yet — DELETE/TRUNCATE would fail
    in that case. The next tasks (load_data, preprocess) recreate each
    table from scratch the moment they write to it.
    """
    hook = MySqlHook(mysql_conn_id=MYSQL_CONN_ID)
    hook.run(f"DROP TABLE IF EXISTS {RAW_TABLE}")
    hook.run(f"DROP TABLE IF EXISTS {PROCESSED_TABLE}")


def load_data():
    """Task 2: loads penguins.csv into MySQL exactly as-is, no cleaning.

    index_col=0 only drops the CSV's leftover unnamed row-index column —
    that's a file-reading detail, not a data preprocessing step. Every
    actual data value (including missing ones, which pandas represents as
    NaN -> SQL NULL) is inserted untouched.
    """
    df = pd.read_csv(CSV_PATH, index_col=0)
    hook = MySqlHook(mysql_conn_id=MYSQL_CONN_ID)
    df.to_sql(RAW_TABLE, con=hook.get_sqlalchemy_engine(), if_exists="replace", index=False)


def preprocess():
    """Task 3: cleans and encodes the raw data for training.

    Drops rows missing any feature or the target (a row with no label
    can't be used for supervised training), then label-encodes the two
    categorical columns. The fitted encoders are saved to the shared
    models/ volume — the training task (and later, the inference API)
    must reuse these exact same encoders, not fit new ones, to stay
    consistent with what the model was trained on.
    """
    hook = MySqlHook(mysql_conn_id=MYSQL_CONN_ID)
    df = hook.get_pandas_df(sql=f"SELECT * FROM {RAW_TABLE}")

    df_clean = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    le_island = LabelEncoder()
    le_sex = LabelEncoder()
    df_clean["island"] = le_island.fit_transform(df_clean["island"])
    df_clean["sex"] = le_sex.fit_transform(df_clean["sex"])

    df_clean.to_sql(PROCESSED_TABLE, con=hook.get_sqlalchemy_engine(), if_exists="replace", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "encoders.pkl", "wb") as f:
        pickle.dump({"island": le_island, "sex": le_sex}, f)


def train_model():
    """Task 4: trains 4 candidate models and saves all of them.

    Same artifact contract as Taller_contenedores — each model as its own
    .pkl, plus a metrics.json with per-model accuracy and a "_best_model"
    key — so the inference API can reuse that project's loading logic
    almost unchanged.
    """
    hook = MySqlHook(mysql_conn_id=MYSQL_CONN_ID)
    df = hook.get_pandas_df(sql=f"SELECT * FROM {PROCESSED_TABLE}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidate_models = {
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics[name] = {"accuracy": round(float(acc), 4)}

        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    best_model = max(metrics, key=lambda name: metrics[name]["accuracy"])
    metrics["_best_model"] = best_model

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


with DAG(
    dag_id="penguins_pipeline",
    schedule=None,
    start_date=datetime(2026, 7, 28),
    catchup=False,
    default_args={
        "retries": 1,
    },
) as dag:
    delete_task = PythonOperator(
        task_id="delete_content",
        python_callable=delete_content,
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    delete_task >> load_task >> preprocess_task >> train_task
