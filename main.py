from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --- TEAMMATE 2: Import the RAG function ---
from rag_engine import get_maintenance_suggestions

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
    
    # --- TEAMMATE 2: Add RAG Logic Here ---
    if prediction[0] == 1:
        result = "Anomaly"
        # Turn the raw numbers into a sentence so the Vector DB can understand it
        query_string = f"Anomaly detected with Temperature {data.Temperature}, Vibration {data.Vibration}, Pressure {data.Pressure}"
        
        # Ask ChromaDB for the 3 best fixes
        suggestions = get_maintenance_suggestions(query_string)
    else:
        result = "Normal"
        suggestions = ["System operating normally. No maintenance required."]
        
    return {
        "input_received": data.dict(),
        "prediction": result,
        "recommended_fixes": suggestions # The API now returns your RAG fixes!
    }