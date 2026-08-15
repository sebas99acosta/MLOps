"""Entrena un DecisionTreeClassifier usando solo bill_length_mm y body_mass_g
para predecir la especie del pingüino, y lo serializa en modelo_penguins.pkl.
"""

import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("penguins.csv", index_col=0)
df_clean = df.dropna(subset=["bill_length_mm", "body_mass_g", "species"])

X = df_clean[["bill_length_mm", "body_mass_g"]]
y = df_clean["species"]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

with open("modelo_penguins.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado en modelo_penguins.pkl")
print("Clases:", list(model.classes_))
