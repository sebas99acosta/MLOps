"""Covertype inference API.

Serves whichever version of the registered model `covertype` carries the
`@champion` alias in MLflow. Everything comes through the MLflow server: this
service has no MinIO credentials. The model is a full sklearn Pipeline, so it
takes raw values such as "Rawah" / "C7202" directly.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version as installed_version

import mlflow
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ.get("MODEL_NAME", "covertype")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

# Libraries whose version must match the one used for training: MLflow only
# warns on a mismatch, and a model loaded under different versions is not
# guaranteed to predict correctly.
CHECKED_LIBRARIES = ["scikit-learn", "numpy", "skops"]

logger = logging.getLogger("uvicorn.error")
client = MlflowClient(MLFLOW_TRACKING_URI)
state = {"model": None, "info": None}


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


def load_champion():
    """Returns (model, info) for the version that currently carries the alias."""
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except MlflowException as error:
        if error.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise NoChampionError(f"No version of '{MODEL_NAME}' has the alias @{MODEL_ALIAS} yet.") from error
        raise

    # Load the resolved version number, not "@champion": if the alias moves
    # while we load, model and info still describe the same version.
    model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
    check_library_versions(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    run = client.get_run(model_version.run_id)
    info = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_version": int(model_version.version),
        "run_id": model_version.run_id,
        "run_name": run.info.run_name,
        "metrics": {name: run.data.metrics.get(name) for name in ("macro_f1", "accuracy")},
        "params": run.data.params,
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return model, info


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A VersionMismatchError here is deliberately NOT caught: the API refuses
    # to start rather than serve a model that may predict wrongly.
    # No champion yet, or MLflow unreachable: start anyway, /predict answers
    # 503 and POST /reload loads the model later.
    try:
        state["model"], state["info"] = load_champion()
        logger.info("Serving %s v%s", MODEL_NAME, state["info"]["model_version"])
    except (NoChampionError, MlflowException, requests.RequestException) as error:
        logger.warning("Starting without a model: %s", error)
    yield


app = FastAPI(title="Covertype inference API (MLflow)", lifespan=lifespan)


class CovertypeFeatures(BaseModel):
    """Raw features, as in processed.covertype. Ranges match the validation
    the notebook applies before training."""

    elevation: int
    aspect: int = Field(ge=0, le=360)
    slope: int
    horizontal_distance_to_hydrology: int
    vertical_distance_to_hydrology: int
    horizontal_distance_to_roadways: int
    hillshade_9am: int = Field(ge=0, le=255)
    hillshade_noon: int = Field(ge=0, le=255)
    hillshade_3pm: int = Field(ge=0, le=255)
    horizontal_distance_to_fire_points: int
    wilderness_area: str = Field(min_length=1)
    soil_type: str = Field(min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "elevation": 2596, "aspect": 51, "slope": 3,
                "horizontal_distance_to_hydrology": 258, "vertical_distance_to_hydrology": 0,
                "horizontal_distance_to_roadways": 510,
                "hillshade_9am": 221, "hillshade_noon": 232, "hillshade_3pm": 148,
                "horizontal_distance_to_fire_points": 6279,
                "wilderness_area": "Rawah", "soil_type": "C7745",
            }]
        }
    }


def mlflow_reachable() -> bool:
    try:
        return requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=2).ok
    except requests.RequestException:
        return False


@app.get("/")
def root():
    info = state["info"]
    return {
        "service": "Covertype inference API (MLflow)",
        "model": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "model_version": info["model_version"] if info else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "mlflow_reachable": mlflow_reachable()}


@app.get("/model")
def current_model():
    """The registry version this API is serving, with its run's metrics."""
    if state["info"] is None:
        raise HTTPException(status_code=404, detail="No model loaded yet.")
    return state["info"]


@app.post("/reload")
def reload_model():
    """Pick up a newly promoted @champion without restarting the container.
    If the new model's library versions don't match, the current one is kept."""
    previous = state["info"]["model_version"] if state["info"] else None
    try:
        model, info = load_champion()
    except VersionMismatchError as error:
        raise HTTPException(status_code=409, detail=f"Kept current model. {error}")
    except NoChampionError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except (MlflowException, requests.RequestException) as error:
        raise HTTPException(status_code=503, detail=f"MLflow not reachable, kept current model: {error}")
    state["model"], state["info"] = model, info
    return {"reloaded": True, "previous_version": previous, "model_version": info["model_version"]}


@app.post("/predict")
def predict(features: CovertypeFeatures):
    model, info = state["model"], state["info"]
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded: train one in Jupyter, then POST /reload.")

    row = pd.DataFrame([features.model_dump()])
    # pyfunc predict() enforces the signature MLflow saved with the model
    # (column names and types): the second line of input validation.
    cover_type = int(model.predict(row)[0])
    sklearn_model = model.get_raw_model()
    probabilities = sklearn_model.predict_proba(row)[0]
    return {
        "cover_type": cover_type,
        "probabilities": {int(c): round(float(p), 4) for c, p in zip(sklearn_model.classes_, probabilities)},
        "model_name": info["model_name"],
        "model_version": info["model_version"],
        "run_id": info["run_id"],
    }
