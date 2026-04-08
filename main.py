from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from rag_engine import get_maintenance_suggestions

app = FastAPI(title="AeroSense IoT Diagnostic API")

# Load the new 8-feature model
model = joblib.load("model.joblib")

# Upgraded Schema to match the new OBD-II sensor data
class SensorData(BaseModel):
    Temperature: float
    Vibration: float
    Pressure: float
    Speed: float
    RPM: float
    Odometer: float
    Battery_Voltage: float
    Outside_Temp: float

@app.post("/predict")
def predict_status(data: SensorData):
    # Pass all 8 features to the AI model
    input_data = [[
        data.Temperature,
        data.Vibration,
        data.Pressure,
        data.Speed,
        data.RPM,
        data.Odometer,
        data.Battery_Voltage,
        data.Outside_Temp
    ]]

    prediction = model.predict(input_data)[0]

    # If the AI detects an anomaly, trigger the ChromaDB RAG Engine
    if prediction == "Anomaly":
        query_text = f"Engine running hot at {data.Temperature}C with intense vibration {data.Vibration} and dying battery {data.Battery_Voltage}V."
        fixes = get_maintenance_suggestions(query_text)
        return {
            "prediction": prediction,
            "recommended_fixes": fixes
        }

    return {
        "prediction": prediction,
        "recommended_fixes": ["System operating normally. Have a safe drive."]
    }