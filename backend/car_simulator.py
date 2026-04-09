import time
import random
import requests

# --- CONFIGURATION ---
# ⚠️ REPLACE THIS WITH YOUR ACTUAL RENDER URL! Keep the /predict at the end.
RENDER_API_URL = "https://aerosense-mlops-pipeline-sh1j.onrender.com/predict" 

def generate_telemetry(scenario):
    """Generates precise physics data based on the chosen scenario."""
    
    # Base baseline for a healthy car
    data = {
        "Temperature": round(random.uniform(88.0, 92.0), 1),
        "Vibration": round(random.uniform(0.4, 0.6), 2),
        "Pressure": round(random.uniform(34.0, 36.0), 1),
        "Speed": round(random.uniform(60.0, 70.0), 1),
        "RPM": round(random.uniform(1800.0, 2200.0), 0),
        "Odometer": round(random.uniform(50000.0, 50100.0), 0),
        "Battery_Voltage": round(random.uniform(14.0, 14.4), 1),
        "Outside_Temp": round(random.uniform(20.0, 25.0), 1)
    }

    # Inject the specific mechanical failures
    if scenario == "Overheating_Risk":
        data["Temperature"] = round(random.uniform(115.0, 122.0), 1)
        data["RPM"] = round(random.uniform(4000.0, 5000.0), 0)
        data["Outside_Temp"] = round(random.uniform(32.0, 38.0), 1)
    
    elif scenario == "Alternator_Failure":
        data["Battery_Voltage"] = round(random.uniform(11.0, 11.6), 1)
        # As voltage drops, engine might run a little rough
        data["Vibration"] = round(random.uniform(0.6, 0.9), 2) 
        
    elif scenario == "Tire_Leak":
        data["Pressure"] = round(random.uniform(22.0, 26.0), 1)
        data["Vibration"] = round(random.uniform(1.0, 1.4), 2)

    return data

def start_engine():
    print("🚗 Starting Advanced OBD-II IoT Simulator...")
    print(f"📡 Target API: {RENDER_API_URL}")
    print("-" * 60)
    
    # The scenarios our simulator can run
    scenarios = ["Normal", "Normal", "Normal", "Overheating_Risk", "Alternator_Failure", "Tire_Leak"]
    trip_counter = 1
    
    while True:
        # Pick a random scenario (weighted heavily towards Normal driving)
        current_scenario = random.choice(scenarios)
        sensor_data = generate_telemetry(current_scenario)
        
        print(f"[{trip_counter}] Driving Mode: {current_scenario}")
        print(f"   ↳ Telemetry: Temp: {sensor_data['Temperature']}C | Vib: {sensor_data['Vibration']} | Press: {sensor_data['Pressure']}PSI | Spd: {sensor_data['Speed']}mph | RPM: {sensor_data['RPM']} | Odo: {sensor_data['Odometer']} | Batt: {sensor_data['Battery_Voltage']}V | Out: {sensor_data['Outside_Temp']}C")
        
        try:
            # Send the data to the cloud
            response = requests.post(RENDER_API_URL, json=sensor_data)
            
            if response.status_code == 200:
                result = response.json()
                cloud_prediction = result['prediction']
                
                if cloud_prediction != 'Normal':
                    print(f"   🚨 CLOUD WARNING: {cloud_prediction} DETECTED!")
                    print(f"   🛠️ AI Advice: {result['recommended_fixes'][0][:75]}...") # Print just the first part of the fix
                else:
                    print("   ✅ Cloud Status: Normal")
            else:
                print(f"   ⚠️ API Error {response.status_code}: Check your Render app!")
                
        except requests.exceptions.RequestException as e:
            print("   ❌ Connection failed! Check internet or URL.")
            
        print("-" * 60)
        trip_counter += 1
        time.sleep(5) 

if __name__ == "__main__":
    start_engine()