import os
import joblib
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AeroSense MLOps API", description="Predictive Maintenance Pipeline")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. PYDANTIC MODEL ---
class TelemetryInput(BaseModel):
    Temperature: float
    Vibration: float
    Pressure: float
    Speed: float
    RPM: float
    Odometer: float
    Battery_Voltage: float
    Outside_Temp: float

# --- 2. LOAD SCIKIT-LEARN MODEL ---
try:
    model = joblib.load("model.joblib")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load ML model. Details: {e}")
    model = None

# --- 3. INITIALIZE CHROMA DB (RAG) ---
try:
    from rag_engine import DB_PATH, COLLECTION_NAME
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    rag_collection = chroma_client.get_collection(name=COLLECTION_NAME)
except ImportError:
    print("CRITICAL ERROR: Could not import DB_PATH or COLLECTION_NAME from rag_engine.py")
    rag_collection = None
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to ChromaDB. Details: {e}")
    rag_collection = None

# Global cache for Ghost Payload detection (In-memory for the demo)
LAST_PAYLOAD_CACHE = None

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict_status(data: TelemetryInput):
    global LAST_PAYLOAD_CACHE
    telemetry_dict = data.dict()
    t_dict = telemetry_dict 
    
    critical_errors = []
    
    # --- NEW FEATURE: 1. The "Ghost Payload" (Telemetry Freeze) ---
    # In the real world, sensors fluctuate. If we get the EXACT same numbers twice, the sensor crashed.
    if LAST_PAYLOAD_CACHE is not None and t_dict == LAST_PAYLOAD_CACHE:
        critical_errors.append("TELEMETRY FREEZE: Sensor array is reporting the exact same values as the previous payload. Possible sensor module crash or network ghost payload.")
    LAST_PAYLOAD_CACHE = t_dict.copy()

    # --- NEW FEATURE: 2. Dynamic Feature Engineering (Gear Ratio) ---
    # We calculate the gear ratio dynamically before inference to catch transmission issues
    if t_dict['Speed'] > 10: # Avoid division by zero and parking lot speeds
        gear_ratio = t_dict['RPM'] / t_dict['Speed']
        # If going 70mph but RPM is 6000 (Ratio > 85), the vehicle is stuck in a low gear
        if t_dict['Speed'] > 60 and gear_ratio > 85:
            critical_errors.append(f"TRANSMISSION ANOMALY: Calculated gear ratio ({gear_ratio:.2f}) is abnormally high for highway speeds ({t_dict['Speed']} mph). Vehicle may be stuck in a lower gear.")

    # --- 3. THE "IMPOSSIBLE" FACTORS (Negative values & Sensor Malfunctions) ---
    if t_dict['RPM'] < 0 or t_dict['RPM'] > 10000:
        critical_errors.append(f"SENSOR FAILURE: RPM reading ({t_dict['RPM']}) is physically impossible.")
    if t_dict['Speed'] < 0 or t_dict['Speed'] > 150:
        critical_errors.append(f"SENSOR FAILURE: Speed reading ({t_dict['Speed']} mph) exceeds mechanical limits.")
    if t_dict['Pressure'] < 0 or t_dict['Pressure'] > 150:
        critical_errors.append(f"SENSOR FAILURE: Oil pressure reading ({t_dict['Pressure']} PSI) indicates a broken transducer.")
    if t_dict['Vibration'] < 0:
        critical_errors.append(f"SENSOR FAILURE: Vibration amplitude ({t_dict['Vibration']}) cannot be negative.")
    if t_dict['Battery_Voltage'] < 0:
        critical_errors.append(f"ELECTRICAL FAULT: Negative voltage ({t_dict['Battery_Voltage']}V) detected. Reversed polarity.")
    if t_dict['Temperature'] < -50 or t_dict['Outside_Temp'] < -60:
        critical_errors.append(f"SENSOR FAILURE: Extreme negative temperature. Thermocouple disconnected.")
    if t_dict['Odometer'] < 0:
        critical_errors.append(f"DATA FRAUD: Negative odometer reading ({t_dict['Odometer']}). Possible ECU tampering or rollback.")

    # --- 4. ABSOLUTE CRITICAL LIMITS ---
    if t_dict['Temperature'] > 115:
        critical_errors.append(f"CRITICAL OVERHEATING: Engine temp ({t_dict['Temperature']}°C) is at structural failure point.")
    if t_dict['Vibration'] > 4.0:
        critical_errors.append(f"CRITICAL VIBRATION: Reading ({t_dict['Vibration']}) indicates severe mechanical imbalance.")
    if 5000 < t_dict['RPM'] <= 10000:
        critical_errors.append(f"REDLINE WARNING: Engine operating at dangerous RPM ({t_dict['RPM']}). Valve float imminent.")
    if t_dict['Battery_Voltage'] > 15.5:
        critical_errors.append(f"ELECTRICAL SURGE: Alternator overcharging ({t_dict['Battery_Voltage']}V). Risk of fire/ECU damage.")

    # --- 5. RELATIONAL ANOMALIES ---
    is_driving = t_dict['Speed'] > 5
    
    # NEW FEATURE: Cold Start Abuse
    if t_dict['Temperature'] < 40 and t_dict['RPM'] > 3500:
        critical_errors.append(f"COLD START ABUSE: High RPM ({t_dict['RPM']}) detected while engine is cold ({t_dict['Temperature']}°C). This causes severe premature engine wear.")

    # Pressure vs. Speed
    if is_driving and t_dict['Pressure'] < 45:
        critical_errors.append(f"OIL STARVATION: Pressure too low ({t_dict['Pressure']} PSI) for driving loads. Metal-on-metal friction occurring.")
    
    # Drivetrain logic
    if t_dict['Speed'] > 60 and t_dict['RPM'] < 1000:
        critical_errors.append(f"DRIVETRAIN MISMATCH: High speed ({t_dict['Speed']}mph) with idle RPM ({t_dict['RPM']}). Clutch slipping or transmission failure.")
    
    # Alternator Logic
    if is_driving and t_dict['Battery_Voltage'] < 12.5:
        critical_errors.append(f"ALTERNATOR FAILURE: Voltage dropping ({t_dict['Battery_Voltage']}V) while driving. Battery will drain shortly.")

    # Thermal Shock
    if t_dict['Temperature'] > 100 and t_dict['Outside_Temp'] < 0:
        critical_errors.append(f"THERMAL SHOCK: Hot engine ({t_dict['Temperature']}°C) in freezing environment ({t_dict['Outside_Temp']}°C). Check coolant mixture.")

    # --- TRIGGER HEURISTIC GUARDRAIL ---
    if critical_errors:
        critical_errors.append("IMMEDIATE DRIVER ACTION: Perform emergency stop or adjust driving behavior as indicated above.")
        return {
            "prediction": "Anomaly",
            "recommended_fixes": critical_errors
        }

    # --- STEP B: Machine Learning Inference ---
    if model is None:
        return {"prediction": "Error", "recommended_fixes": ["ML Model failed to load on server."]}

    # Extract features for scikit-learn
    features = [[
        t_dict["Temperature"],
        t_dict["Vibration"],
        t_dict["Pressure"],
        t_dict["Speed"],
        t_dict["RPM"],
        t_dict["Odometer"],
        t_dict["Battery_Voltage"],
        t_dict["Outside_Temp"]
    ]]
    
    prediction_result = model.predict(features)[0] 

    # --- STEP C: Contextual AI (RAG) Retrieval ---
    if prediction_result == 1 or prediction_result == "Anomaly": 
        fixes = ["General maintenance required. Inspect system."] 
        
        if rag_collection is not None:
            query_string = f"High temperature {t_dict['Temperature']} or high vibration {t_dict['Vibration']} anomaly."
            rag_results = rag_collection.query(
                query_texts=[query_string],
                n_results=3 
            )
            
            if rag_results['documents'] and len(rag_results['documents'][0]) > 0:
                fixes = rag_results['documents'][0]

        return {
            "prediction": "Anomaly",
            "recommended_fixes": fixes
        }

    # --- STEP D: Normal Return ---
    return {
        "prediction": "Normal",
        "recommended_fixes": ["System healthy. All telemetry within normal parameters."]
    }

# --- HEALTH CHECK ENDPOINT ---
@app.get("/")
async def root():
    return {"status": "AeroSense API is running."}