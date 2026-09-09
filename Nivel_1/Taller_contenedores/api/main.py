"""API de inferencia para el clasificador de especies de pingüinos.

Los modelos se entrenan en un servicio separado (JupyterLab) y se guardan en
un volumen compartido (/models). Este API observa ese volumen con watchdog:
en cuanto el notebook guarda un modelo nuevo, el API lo detecta y lo agrega
a su registro en memoria sin necesidad de reiniciar el contenedor.
"""

import json
import pickle
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

MODELS_DIR = Path("/models")

app = FastAPI(title="Penguins Species Classifier API")

registry_lock = threading.Lock()
state = {
    "models": {},      # nombre -> objeto modelo cargado
    "metrics": {},      # nombre -> {"accuracy": ...}
    "encoders": None,   # {"island": LabelEncoder, "sex": LabelEncoder}
    "current_model": None,
}


class PenguinFeatures(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str


class ModelSelection(BaseModel):
    model_name: str


# --- Registro de modelos: carga y recarga desde el volumen compartido -----

def load_registry():
    """Relee /models desde disco y reconstruye el registro en memoria.

    Se llama al iniciar el API y cada vez que watchdog detecta cambios en
    el volumen compartido (nuevo modelo guardado desde el notebook).
    """
    metrics_path = MODELS_DIR / "metrics.json"
    encoders_path = MODELS_DIR / "encoders.pkl"

    if not metrics_path.exists():
        # Aún no se ha entrenado nada; el API sigue esperando.
        return

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_model = metrics.pop("_best_model", None)

        models = {}
        for name in metrics:
            model_path = MODELS_DIR / f"{name}.pkl"
            if not model_path.exists():
                continue
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)

        encoders = None
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                encoders = pickle.load(f)

        with registry_lock:
            state["models"] = models
            state["metrics"] = metrics
            state["encoders"] = encoders
            # Si el modelo activo ya no existe, se cae al mejor disponible.
            if state["current_model"] not in models:
                state["current_model"] = best_model if best_model in models else next(
                    iter(models), None
                )

        print(f"[registry] recargado: {list(models.keys())} "
              f"(activo={state['current_model']})")
    except Exception as e:
        # Un pickle a medio escribir o un JSON incompleto no debe tumbar el
        # API: se ignora y se reintenta en el próximo evento de watchdog.
        print(f"[registry] error recargando, se reintentará: {e}")


class DebouncedReloadHandler(FileSystemEventHandler):
    """Agrupa ráfagas de eventos (varios archivos guardados juntos) en una
    sola recarga, dando tiempo a que terminen de escribirse en disco."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self._timer = None
        self._timer_lock = threading.Lock()

    def _schedule_reload(self):
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, load_registry)
            self._timer.daemon = True
            self._timer.start()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_reload()

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_reload()

    def on_moved(self, event):
        # Un guardado atómico (escribir a .tmp y luego os.replace) dispara
        # este evento en vez de on_created/on_modified.
        self._schedule_reload()


@app.on_event("startup")
def start_watchdog():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_registry()  # por si ya había modelos guardados de una corrida previa

    handler = DebouncedReloadHandler(delay=1.0)
    observer = Observer()
    observer.schedule(handler, str(MODELS_DIR), recursive=False)
    observer.daemon = True
    observer.start()
    app.state.observer = observer


@app.on_event("shutdown")
def stop_watchdog():
    observer = getattr(app.state, "observer", None)
    if observer:
        observer.stop()
        observer.join()


# --- Endpoints --------------------------------------------------------------

@app.get("/")
def root():
    with registry_lock:
        return {
            "message": "Penguins Species Classifier API",
            "current_model": state["current_model"],
            "available_models": list(state["models"].keys()),
        }


@app.get("/models")
def list_models():
    with registry_lock:
        return {
            "current_model": state["current_model"],
            "models": state["metrics"],
        }


@app.post("/reload-models")
def reload_models():
    """Recarga manual, por si se necesita forzar sin esperar a watchdog."""
    load_registry()
    with registry_lock:
        return {
            "current_model": state["current_model"],
            "models": list(state["models"].keys()),
        }


@app.post("/select-model")
def select_model(selection: ModelSelection):
    with registry_lock:
        if selection.model_name not in state["models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{selection.model_name}' no existe. "
                       f"Opciones: {list(state['models'].keys())}",
            )
        state["current_model"] = selection.model_name
        return {"current_model": state["current_model"]}


@app.post("/predict")
def predict_species(data: PenguinFeatures):
    with registry_lock:
        current = state["current_model"]
        encoders = state["encoders"]
        models = state["models"]

    if not current or current not in models or encoders is None:
        raise HTTPException(
            status_code=503,
            detail="Todavía no hay modelos entrenados disponibles.",
        )

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

    prediction = models[current].predict(input_data)

    return {
        "predicted_species": str(prediction[0]),
        "model_used": current,
    }
