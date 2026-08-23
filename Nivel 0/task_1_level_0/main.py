import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Esquema de entrada: valida que la petición traiga exactamente estos campos.
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    body_mass_g: float


# Cargar el modelo en memoria al iniciar
with open("modelo_penguins.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict_species(data: PenguinFeatures):
    # Formatear datos
    input_data = [[data.bill_length_mm, data.body_mass_g]]

    # Inferencia
    prediction = model.predict(input_data)

    # Retornar JSON
    return {"predicted_species": str(prediction[0])}
