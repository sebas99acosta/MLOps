"""Covertype inference API.

Serves whichever model covertype/production.json in MinIO points to.
The model is a full sklearn Pipeline, so it takes raw values such as
"Rawah" / "C7202" directly — no encoding happens here.
"""

import json
import os
import pickle
from contextlib import asynccontextmanager

import boto3
import numpy
import pandas as pd
import sklearn
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MINIO_BUCKET = os.environ["MINIO_BUCKET"]
PRODUCTION_POINTER_KEY = "covertype/production.json"

# Libraries whose version must match the one used for training: a pickled
# sklearn Pipeline (and the numpy arrays inside it) is not guaranteed to load
# correctly under different versions.
CHECKED_LIBRARIES = {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__}

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["MINIO_ENDPOINT"],
    aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
    aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
)

state = {"model": None, "pointer": None}


class VersionMismatchError(RuntimeError):
    pass


def check_library_versions(trained_with: dict):
    mismatches = {
        lib: {"trained_with": trained_with.get(lib), "installed": installed}
        for lib, installed in CHECKED_LIBRARIES.items()
        if trained_with.get(lib) != installed
    }
    if mismatches:
        raise VersionMismatchError(f"Model was trained with different library versions: {mismatches}")


def load_production_model():
    """Returns (model, pointer), or (None, None) if nothing is in production yet."""
    try:
        pointer = json.loads(
            s3.get_object(Bucket=MINIO_BUCKET, Key=PRODUCTION_POINTER_KEY)["Body"].read()
        )
    except ClientError as error:
        if error.response["Error"]["Code"] == "NoSuchKey":
            return None, None
        raise

    check_library_versions(pointer["library_versions"])
    model = pickle.loads(s3.get_object(Bucket=MINIO_BUCKET, Key=pointer["model_key"])["Body"].read())
    return model, pointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A VersionMismatchError here is deliberately NOT caught: the API refuses
    # to start rather than serve a model that may predict wrongly.
    # A missing model is fine: the API starts and /predict answers 503.
    state["model"], state["pointer"] = load_production_model()
    yield


app = FastAPI(title="Covertype inference API", lifespan=lifespan)


class CovertypeFeatures(BaseModel):
    """Raw features, as the Data API delivers them. Ranges match the
    validation the Airflow DAG applies before training."""

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
                "elevation": 2906, "aspect": 91, "slope": 18,
                "horizontal_distance_to_hydrology": 579, "vertical_distance_to_hydrology": 95,
                "horizontal_distance_to_roadways": 1827,
                "hillshade_9am": 244, "hillshade_noon": 209, "hillshade_3pm": 88,
                "horizontal_distance_to_fire_points": 2782,
                "wilderness_area": "Commanche", "soil_type": "C7202",
            }]
        }
    }


@app.get("/")
def root():
    return {
        "service": "Covertype inference API",
        "model_version": state["pointer"]["version"] if state["pointer"] else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model")
def current_model():
    """The production pointer this API is currently serving."""
    if state["pointer"] is None:
        raise HTTPException(status_code=404, detail="No model in production yet.")
    return state["pointer"]


@app.post("/reload")
def reload_model():
    """Pick up a newly promoted model without restarting the container.
    If the new model's library versions don't match, the current one is kept."""
    try:
        model, pointer = load_production_model()
    except VersionMismatchError as error:
        raise HTTPException(status_code=409, detail=f"Kept current model. {error}")
    if model is None:
        raise HTTPException(status_code=404, detail="No model in production yet.")
    state["model"], state["pointer"] = model, pointer
    return {"reloaded": True, "model_version": pointer["version"]}


@app.post("/predict")
def predict(features: CovertypeFeatures):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="No model in production yet: train one in Jupyter first.")

    row = pd.DataFrame([features.model_dump()])
    probabilities = model.predict_proba(row)[0]
    return {
        "cover_type": int(model.predict(row)[0]),
        "model_version": state["pointer"]["version"],
        "probabilities": {int(c): round(float(p), 4) for c, p in zip(model.classes_, probabilities)},
    }
