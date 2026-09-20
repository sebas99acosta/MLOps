"""Inference API for the penguins species classifier.

Loads whatever train_model (the last task of the Airflow DAG) left in the
shared models/ volume: one .pkl per candidate model, metrics.json (with
each model's accuracy and the best one), and encoders.pkl. Same artifact
contract as the Taller_contenedores project, so this loading logic is
reused almost unchanged.

Unlike Taller_contenedores, there's no watchdog here: models are only
produced once per DAG run, not continuously while this API is serving
requests, so a single load at startup is enough.
"""

import json
import pickle
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path("/app/models")

app = FastAPI(title="Penguins Species Classifier API (Airflow-trained)")


class ModelName(str, Enum):
    """Fixed set of models train_model can produce. Used as a query
    parameter (not a request-body field) specifically so FastAPI's /docs
    UI renders it as a dropdown — Swagger UI only does that for
    query/path parameters, never for fields inside a JSON body. Adding a
    5th candidate model means editing both the DAG and this Enum
    together, which is expected: the DAG is the only thing that decides
    what could ever exist in models/ in the first place.
    """

    decision_tree = "decision_tree"
    random_forest = "random_forest"
    knn = "knn"
    logistic_regression = "logistic_regression"


class PenguinFeatures(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str


# --- Load artifacts once at startup -----------------------------------

with open(MODELS_DIR / "metrics.json") as f:
    metrics = json.load(f)

best_model_name = metrics.pop("_best_model")

models = {}
for name in metrics:
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        models[name] = pickle.load(f)

with open(MODELS_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

state = {"current_model": best_model_name}


# --- Endpoints -----------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Penguins Species Classifier API",
        "current_model": state["current_model"],
        "available_models": list(models.keys()),
    }


@app.get("/models")
def list_models():
    return {
        "current_model": state["current_model"],
        "models": metrics,
    }


@app.post("/select-model")
def select_model(model_name: ModelName):
    state["current_model"] = model_name.value
    return {"current_model": state["current_model"]}


@app.post("/predict")
def predict_species(data: PenguinFeatures):
    try:
        island_enc = encoders["island"].transform([data.island])[0]
        sex_enc = encoders["sex"].transform([data.sex])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    input_data = [[
        island_enc,
        data.bill_length_mm,
        data.bill_depth_mm,
        data.flipper_length_mm,
        data.body_mass_g,
        sex_enc,
    ]]

    model = models[state["current_model"]]
    prediction = model.predict(input_data)

    return {
        "predicted_species": str(prediction[0]),
        "model_used": state["current_model"],
    }
