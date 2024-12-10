from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle
model = joblib.load("best_rf_model.pkl")

# Schéma de la requête
class PredictRequest(BaseModel):
    input: list

@app.post("/predict")
def predict(request: PredictRequest):
    data = np.array(request.input).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}