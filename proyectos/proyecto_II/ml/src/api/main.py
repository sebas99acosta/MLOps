"""Diabetes readmission inference API.

Serves whichever version of the registered model `diabetes_readmission`
carries the `@champion` alias in MLflow. Airflow moves that alias when a
retrained model wins, and this API follows it with no code change or restart:

- at startup it loads the champion;
- every RELOAD_INTERVAL_SECONDS a background thread asks MLflow which version
  the alias points to (one cheap metadata call) and only downloads the model
  when the version changed;
- POST /reload forces that same check right away.

Input is the raw encounter, as in the source CSV ("[70-80)", ICD-9 codes, "?"
for unknown). The API applies `diabetes_features.clean_features`, the same
code the training pipeline used, so the model always receives exactly what it
was trained on. Every prediction is written to `inference.predictions` after
the response is sent (BackgroundTasks), so the database adds no latency.
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version as installed_version
from typing import Annotated, Literal

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import text

from diabetes_features import clean_features
from trainer.common import CHAMPION_ALIAS, MODEL_NAME, db_engine

RELOAD_INTERVAL_SECONDS = int(os.environ.get("RELOAD_INTERVAL_SECONDS", "60"))

# Libraries whose version must match the one used for training: MLflow only
# warns on a mismatch. Training and serving share one image, so this only
# fails if someone serves a model trained elsewhere.
CHECKED_LIBRARIES = ["scikit-learn", "numpy", "skops"]

logger = logging.getLogger("uvicorn.error")
client = MlflowClient()  # MLFLOW_TRACKING_URI from the environment
engine = db_engine()

# The served model. Replaced as a whole (one assignment), so a request always
# sees a model and the info that belongs to it.
state = {"model": None, "info": None, "last_check": None, "last_check_error": None}
reload_lock = threading.Lock()  # one load at a time (background thread vs /reload)

# --- Prometheus metrics (one uvicorn worker per pod, so one registry). ---
REQUESTS = Counter("api_predict_requests_total", "Calls to /predict by HTTP status code", ["status"])
LATENCY = Histogram(
    "api_predict_latency_seconds", "Time to answer /predict (cleaning + model)",
    # Up to 60 s: under overload requests wait seconds for a thread, and a
    # histogram that tops out at 5 s would report "5 s" for all of them.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60),
)
PREDICTIONS = Counter("api_predictions_total", "Predictions by class and model version",
                      ["prediction", "model_version"])
PROBABILITY = Histogram("api_predicted_probability", "Predicted P(readmitted < 30 days)",
                        buckets=[i / 10 for i in range(1, 11)])
INFERENCE_LOG_ERRORS = Counter("api_inference_log_errors_total", "Predictions that could not be logged")
MODEL_VERSION = Gauge("api_model_version", "Registry version currently served (0 = none)")
MODEL_RELOADS = Counter("api_model_reloads_total", "Champion changes picked up", ["trigger"])


class VersionMismatchError(RuntimeError):
    pass


class NoChampionError(LookupError):
    pass


def check_library_versions(model_uri: str):
    """Compare the model's recorded requirements.txt with what is installed here."""
    requirements_path = mlflow.pyfunc.get_model_dependencies(model_uri)
    with open(requirements_path) as file:
        trained_with = dict(
            line.split(";")[0].strip().split("==", 1) for line in file if "==" in line
        )
    mismatches = {
        library: {"trained_with": trained_with.get(library), "installed": installed_version(library)}
        for library in CHECKED_LIBRARIES
        if trained_with.get(library) != installed_version(library)
    }
    if mismatches:
        raise VersionMismatchError(f"Model was trained with different library versions: {mismatches}")


def champion_version():
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as error:
        if error.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise NoChampionError(f"No version of '{MODEL_NAME}' has the alias @{CHAMPION_ALIAS} yet.") from error
        raise


def load_version(model_version):
    """Returns (model, info) for one registry version. The decision threshold
    and metrics travel with the version as tags (set by Airflow's promote step)."""
    # Load the resolved version number, not "@champion": if the alias moves
    # while we load, model and info still describe the same version.
    model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
    check_library_versions(model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    tags = model_version.tags
    info = {
        "model_name": MODEL_NAME,
        "model_alias": CHAMPION_ALIAS,
        "model_version": str(model_version.version),
        "run_id": model_version.run_id,
        "run_name": tags.get("run_name"),
        "model_type": tags.get("model_type"),
        "decision_threshold": float(tags["decision_threshold"]),
        "metrics": {name: float(tags[name]) for name in
                    ("validation_pr_auc", "validation_recall", "validation_precision", "test_pr_auc")
                    if name in tags},
        "trained_by_dag_run": tags.get("dag_run_id"),
        "promotion": tags.get("comparison"),
        "registered_at": datetime.fromtimestamp(model_version.creation_timestamp / 1000, timezone.utc)
                                 .isoformat(timespec="seconds"),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return model, info


def refresh_champion(trigger: str) -> bool:
    """Load the champion if it is not the version already served. Returns
    True if the served model changed. Errors propagate; the current model is
    kept whatever happens."""
    with reload_lock:
        state["last_check"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        model_version = champion_version()
        current = state["info"]["model_version"] if state["info"] else None
        if str(model_version.version) == current:
            return False
        model, info = load_version(model_version)
        state["model"], state["info"] = model, info  # swapped together
        MODEL_VERSION.set(int(info["model_version"]))
        MODEL_RELOADS.labels(trigger=trigger).inc()
        logger.info("Serving %s v%s (was v%s, trigger: %s)", MODEL_NAME, info["model_version"], current, trigger)
        return True


def watch_champion():
    """Background thread: follow @champion. A failed check (MLflow down, no
    champion, version mismatch) only logs; the current model keeps serving."""
    while True:
        time.sleep(RELOAD_INTERVAL_SECONDS)
        try:
            refresh_champion("poll")
            state["last_check_error"] = None
        except Exception as error:  # noqa: BLE001 - the watcher must never die
            state["last_check_error"] = str(error)
            logger.warning("Champion check failed, keeping current model: %s", error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No champion yet or MLflow unreachable: start anyway, /health answers 503
    # (pod not ready, gets no traffic) until the watcher loads a model.
    try:
        refresh_champion("startup")
    except Exception as error:  # noqa: BLE001
        state["last_check_error"] = str(error)
        logger.warning("Starting without a model: %s", error)
    MODEL_VERSION.set(int(state["info"]["model_version"]) if state["info"] else 0)
    threading.Thread(target=watch_champion, name="champion-watcher", daemon=True).start()
    yield


app = FastAPI(
    title="Diabetes readmission API",
    description="Risk of readmission within 30 days for a diabetic inpatient encounter. "
                "Serves the MLflow model `diabetes_readmission@champion`.",
    lifespan=lifespan,
)


@app.middleware("http")
async def count_predict_responses(request, call_next):
    """Count every /predict answer by status code, including the ones the
    handler never sees: 422 (rejected by validation) and 500 (unexpected error)."""
    if request.url.path != "/predict":
        return await call_next(request)
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(status="500").inc()
        raise
    REQUESTS.labels(status=str(response.status_code)).inc()
    return response

# --- Input: the raw encounter, as in the source CSV. ---
AgeBracket = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
Medication = Literal["No", "Steady", "Up", "Down"]
Count = Annotated[int, Field(ge=0)]


class Encounter(BaseModel):
    """One hospital encounter. Categorical values not seen in training (or "?")
    are accepted: the model groups them with the infrequent/unknown ones."""

    race: str = Field(description='e.g. "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"')
    gender: Literal["Female", "Male"]
    age: AgeBracket
    admission_type_id: int = Field(ge=1, description="Code from IDS_mapping.csv, e.g. 1 = Emergency")
    discharge_disposition_id: int = Field(ge=1, description="1 = discharged home. 11/19/20/21 (expired) "
                                                            "and 13/14 (hospice) are outside the model's scope")
    admission_source_id: int = Field(ge=1, description="e.g. 7 = Emergency Room")
    time_in_hospital: int = Field(ge=1, le=14, description="Days")
    medical_specialty: str = Field(description='Admitting physician specialty, e.g. "InternalMedicine", "?"')
    num_lab_procedures: Count
    num_procedures: Count
    num_medications: Count
    number_outpatient: Count
    number_emergency: Count
    number_inpatient: Count
    diag_1: str = Field(description='Primary diagnosis, ICD-9 code, e.g. "428", "250.83", "V57", "?"')
    diag_2: str = Field(description="Secondary diagnosis, ICD-9 code")
    diag_3: str = Field(description="Additional secondary diagnosis, ICD-9 code")
    number_diagnoses: int = Field(ge=1)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = Field(description='"None" = test not performed')
    a1cresult: Literal["None", "Norm", ">7", ">8"] = Field(description='"None" = test not performed')
    metformin: Medication = "No"
    repaglinide: Medication = "No"
    nateglinide: Medication = "No"
    glimepiride: Medication = "No"
    glipizide: Medication = "No"
    glyburide: Medication = "No"
    pioglitazone: Medication = "No"
    rosiglitazone: Medication = "No"
    acarbose: Medication = "No"
    insulin: Medication = "No"
    glyburide_metformin: Medication = "No"
    change: Literal["No", "Ch"] = Field(description='"Ch" = diabetes medication changed')
    diabetesmed: Literal["No", "Yes"] = Field(description="Any diabetes medication prescribed")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "race": "Caucasian", "gender": "Female", "age": "[70-80)",
                "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
                "time_in_hospital": 5, "medical_specialty": "InternalMedicine",
                "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18,
                "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2,
                "diag_1": "428", "diag_2": "250.02", "diag_3": "403", "number_diagnoses": 9,
                "max_glu_serum": "None", "a1cresult": ">8",
                "metformin": "No", "insulin": "Up", "change": "Ch", "diabetesmed": "Yes",
            }]
        }
    }


class Prediction(BaseModel):
    request_id: str
    prediction: int = Field(description="1 = high risk of readmission within 30 days")
    label: str
    probability: float = Field(description="P(readmitted < 30 days)")
    decision_threshold: float
    model_name: str
    model_version: str
    model_alias: str
    latency_ms: float


def log_inference(row: dict) -> None:
    """Runs after the response is sent. A failed write is counted, not raised:
    the client already has its prediction."""
    try:
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO inference.predictions (request_id, input, prediction, probability, threshold, "
                     "model_name, model_version, model_alias, latency_ms) VALUES (:request_id, "
                     "CAST(:input AS jsonb), :prediction, :probability, :threshold, :model_name, "
                     ":model_version, :model_alias, :latency_ms)"),
                {**row, "input": json.dumps(row["input"])},
            )
    except Exception as error:  # noqa: BLE001
        INFERENCE_LOG_ERRORS.inc()
        logger.error("Could not log inference %s: %s", row["request_id"], error)


# The light endpoints are `async def`: they run on the event loop, not in the
# worker thread pool that /predict (a blocking `def`) uses. Under overload that
# pool is full, and probes and Prometheus scrapes queued behind predictions time
# out -> the pod is marked not ready and drops out of the Service (measured with
# Locust at 200 users). They only read in-memory state, so they never block.
@app.get("/")
async def root():
    info = state["info"]
    return {
        "service": "Diabetes readmission API",
        "model": f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}",
        "model_version": info["model_version"] if info else None,
        "docs": "/docs",
    }


@app.get("/health")
async def health(response: Response):
    """Readiness: 200 only when a model is loaded (Kubernetes sends no traffic otherwise)."""
    loaded = state["model"] is not None
    if not loaded:
        response.status_code = 503
    return {"status": "ok" if loaded else "no model loaded",
            "model_version": state["info"]["model_version"] if loaded else None}


@app.get("/model-info")
async def model_info():
    """The registry version being served: threshold, metrics, and how it was promoted."""
    if state["info"] is None:
        raise HTTPException(status_code=503, detail="No model loaded yet.")
    return {**state["info"], "reload_interval_seconds": RELOAD_INTERVAL_SECONDS,
            "last_check": state["last_check"], "last_check_error": state["last_check_error"]}


@app.post("/reload")
def reload_model():
    """Check @champion now instead of waiting for the next poll."""
    previous = state["info"]["model_version"] if state["info"] else None
    try:
        changed = refresh_champion("manual")
    except VersionMismatchError as error:
        raise HTTPException(status_code=409, detail=f"Kept current model. {error}")
    except NoChampionError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except MlflowException as error:
        raise HTTPException(status_code=503, detail=f"MLflow error, kept current model: {error}")
    return {"reloaded": changed, "previous_version": previous, "model_version": state["info"]["model_version"]}


@app.post("/predict", response_model=Prediction)
def predict(encounter: Encounter, background_tasks: BackgroundTasks):
    started = time.perf_counter()
    model, info = state["model"], state["info"]  # same version for the whole request
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded: run the Airflow DAG to promote one.")

    raw = encounter.model_dump()
    features = clean_features(pd.DataFrame([raw]))
    probability = float(model.predict_proba(features)[0, 1])
    threshold = info["decision_threshold"]
    prediction = int(probability >= threshold)
    latency = time.perf_counter() - started

    result = Prediction(
        request_id=str(uuid.uuid4()),
        prediction=prediction,
        label="readmission <30 days: HIGH risk" if prediction else "readmission <30 days: low risk",
        probability=round(probability, 4),
        decision_threshold=threshold,
        model_name=info["model_name"],
        model_version=info["model_version"],
        model_alias=info["model_alias"],
        latency_ms=round(latency * 1000, 2),
    )
    background_tasks.add_task(log_inference, {
        "request_id": result.request_id, "input": raw, "prediction": prediction,
        "probability": probability, "threshold": threshold, "model_name": info["model_name"],
        "model_version": info["model_version"], "model_alias": info["model_alias"],
        "latency_ms": result.latency_ms,
    })

    LATENCY.observe(latency)
    PREDICTIONS.labels(prediction=str(prediction), model_version=info["model_version"]).inc()
    PROBABILITY.observe(probability)
    return result


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
