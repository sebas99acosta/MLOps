# Taller 1 — Clasificador de pingüinos: entrenamiento, API y contenedor

Pipeline completo: procesamiento de datos + entrenamiento de varios modelos
candidatos (`train.py`), un API de inferencia con FastAPI (`main.py`), y un
Dockerfile que empaqueta el API junto con los modelos ya entrenados.

## Arquitectura

- **`train.py`** corre en la máquina de desarrollo. Lee `penguins.csv`,
  limpia y codifica los datos, entrena varios modelos candidatos
  (Decision Tree, Random Forest, KNN, Logistic Regression), evalúa cada
  uno con accuracy sobre un conjunto de prueba, y guarda todos los
  artefactos en `models/` (`*.pkl`, `metrics.json`, `encoders.pkl`).
- **`main.py`** es el API de inferencia. Carga todos los modelos de
  `models/` al iniciar y expone endpoints para predecir, listar modelos
  disponibles, y elegir cuál usar.
- **`Dockerfile`** empaqueta solo el API y los modelos ya entrenados
  (no incluye `train.py` ni `penguins.csv`) — el entrenamiento se hace
  fuera del contenedor, como en el diagrama del taller.

## Uso

### 1. Entrenar (genera `models/`)

```bash
pip install -r requirements.txt
python train.py
```

### 2. Construir la imagen

```bash
docker build -t penguins-api .
```

### 3. Correr el contenedor (API en el puerto 8025)

```bash
docker run -d -p 8025:8025 --name penguins-api penguins-api
```

## Endpoints

- `GET /` — info general y modelo actualmente seleccionado.
- `GET /models` — lista de modelos disponibles con su accuracy.
- `POST /select-model` — logistic_regression, knn, random_forest, decision_tree.
  ```json
  {"model_name": "knn"}
  ```
- `POST /predict` — predice la especie con el modelo seleccionado.
  ```json
  {
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "male"
  }
  ```
  Respuesta:
  ```json
  {"predicted_species": "Adelie", "model_used": "knn"}
  ```

Documentación interactiva en `http://localhost:8025/docs`.
