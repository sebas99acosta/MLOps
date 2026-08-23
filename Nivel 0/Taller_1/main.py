"""API de inferencia para el clasificador de especies de pingüinos.

Carga todos los modelos entrenados por train.py (guardados en models/) y
expone un endpoint de predicción que usa el modelo actualmente seleccionado,
además de endpoints para listar los modelos disponibles y cambiar cuál se
usa para inferencia.
"""

import json
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path("models")

app = FastAPI(title="Penguins Species Classifier API")


class PenguinFeatures(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str


class ModelSelection(BaseModel):
    model_name: str


# --- Carga de artefactos al iniciar --------------------------------------

with open(MODELS_DIR / "metrics.json") as f:
    metrics = json.load(f)

best_model_name = metrics.pop("_best_model")

models = {}
for name in metrics:
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        models[name] = pickle.load(f)

with open(MODELS_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Modelo activo para inferencia; por defecto el de mejor accuracy en test.
state = {"current_model": best_model_name}


# --- Endpoints -------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Penguins Species Classifier API",
        "current_model": state["current_model"],
        "available_models": list(models.keys()),
    }


@app.get("/models")
def list_models():
    """Lista los modelos disponibles, su accuracy y cuál está seleccionado."""
    return {
        "current_model": state["current_model"],
        "models": metrics,
    }


@app.post("/select-model")
def select_model(selection: ModelSelection):
    """Bono: permite elegir con qué modelo se hace la inferencia."""
    if selection.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{selection.model_name}' no existe. "
                   f"Opciones: {list(models.keys())}",
        )
    state["current_model"] = selection.model_name
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
