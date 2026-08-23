"""
Etapa 1: Procesamiento de datos
Etapa 2: Creación de modelo

Lee penguins.csv, lo limpia y codifica, entrena varios modelos candidatos
para clasificar la especie, y guarda cada modelo entrenado junto con sus
métricas en la carpeta models/. El API (main.py) carga todos estos modelos
y permite elegir con cuál hacer inferencia.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODELS_DIR = Path("models")
FEATURE_COLS = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
TARGET_COL = "species"

# --- Etapa 1: Procesamiento de datos -----------------------------------

# 1.1 Carga de datos
df = pd.read_csv("penguins.csv", index_col=0)

# 1.2 Limpieza: se eliminan filas con valores nulos en las columnas usadas
df_clean = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

# 1.3 Codificación de variables categóricas
le_island = LabelEncoder()
le_sex = LabelEncoder()
df_clean["island"] = le_island.fit_transform(df_clean["island"])
df_clean["sex"] = le_sex.fit_transform(df_clean["sex"])

X = df_clean[FEATURE_COLS]
y = df_clean[TARGET_COL]

# 1.4 División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Etapa 2: Creación de modelo ----------------------------------------

# 2.1 Modelos candidatos: la experimentación suele arrojar varios modelos
# con buen desempeño, por eso se entrenan y guardan todos.
candidate_models = {
    "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    # KNN y regresión logística son sensibles a la escala de las variables
    # (body_mass_g en miles vs. bill_length_mm en decenas), por eso van en
    # un pipeline con StandardScaler.
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
}

MODELS_DIR.mkdir(exist_ok=True)
metrics = {}

for name, model in candidate_models.items():
    # 2.2 Entrenamiento
    model.fit(X_train, y_train)

    # 2.3 Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics[name] = {"accuracy": round(float(acc), 4)}
    print(f"{name}: accuracy={acc:.4f}")

    # 2.4 Guardado del modelo
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# El mejor modelo según accuracy queda marcado como selección por defecto
best_model = max(metrics, key=lambda name: metrics[name]["accuracy"])
metrics["_best_model"] = best_model

with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Encoders necesarios para transformar island/sex en el API
with open(MODELS_DIR / "encoders.pkl", "wb") as f:
    pickle.dump({"island": le_island, "sex": le_sex}, f)

print(f"\nMejor modelo: {best_model} (accuracy={metrics[best_model]['accuracy']})")
print(f"Artefactos guardados en {MODELS_DIR.resolve()}")
