# MLOps — Maestría PUJ

Repositorio de trabajos del curso de MLOps.

## Contenido

| Carpeta | Descripción |
|---|---|
| [`task_1_level_0/`](task_1_level_0/) | Clasificación de pingüinos con Árbol de Decisión + API de inferencia con FastAPI |

## task_1_level_0

Entrena un `DecisionTreeClassifier` sobre el dataset `penguins.csv` para predecir la
especie del pingüino, y expone el modelo a través de un endpoint REST con FastAPI.

### Archivos

- `penguins_decision_tree.ipynb` — notebook de exploración, entrenamiento y evaluación (usa las 6 features).
- `train_model.py` — entrena el modelo servido por la API (usa `bill_length_mm` y `body_mass_g`) y genera `modelo_penguins.pkl`.
- `main.py` — aplicación FastAPI con el endpoint `POST /predict`.
- `requirements.txt` — dependencias.

### Uso

```bash
cd task_1_level_0
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Documentación interactiva en http://127.0.0.1:8000/docs

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 45.0, "body_mass_g": 5000}'
```

Respuesta:

```json
{"predicted_species": "Gentoo"}
```
