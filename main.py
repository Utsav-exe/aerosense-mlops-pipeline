from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    print("Error: Model not found. Run train_model.py first!")

app = FastAPI(title="AeroSense Predictive Maintenance API")

class SensorData(BaseModel):
    Temperature: float
    Vibration: float
    Pressure: float

@app.post("/predict")
def predict_status(data: SensorData):
    input_data = np.array([[data.Temperature, data.Vibration, data.Pressure]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Anomaly"
    else:
        result = "Normal"

    return {
        "input_received": data.dict(),
        "prediction": result
    }