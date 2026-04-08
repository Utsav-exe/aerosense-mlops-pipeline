from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from rag_engine import get_maintenance_suggestions

app = FastAPI(title="AeroSense IoT Diagnostic API")

# This tells the API to allow our future web dashboard to talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all web domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],  # Allows all headers
)

# Load the new 8-feature model
model = joblib.load("model.joblib")

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

    # If the AI detects anything other than "Normal", trigger the RAG Engine
    if prediction != "Normal":
        # We give the RAG engine the specific failure type so it gives highly accurate advice!
        query_text = f"Vehicle diagnosed with {prediction}. Current sensors: Temp {data.Temperature}C, Battery {data.Battery_Voltage}V, Pressure {data.Pressure}PSI."
        fixes = get_maintenance_suggestions(query_text)
        return {
            "prediction": prediction,
            "recommended_fixes": fixes
        }

    return {
        "prediction": prediction,
        "recommended_fixes": ["System healthy. All telemetry within normal parameters."]
    }