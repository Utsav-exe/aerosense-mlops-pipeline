import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Generating 8-dimensional IoT vehicle data...")

# 1. Generate "Normal" driving data
normal_data = pd.DataFrame({
    'Temperature': np.random.normal(90, 5, 500),      # Normal engine temp (90C)
    'Vibration': np.random.normal(0.5, 0.1, 500),     # Smooth ride
    'Pressure': np.random.normal(35, 2, 500),         # Good tire pressure
    'Speed': np.random.normal(65, 10, 500),           # Highway cruising (65 mph)
    'RPM': np.random.normal(2000, 300, 500),          # Normal RPM
    'Odometer': np.random.normal(50000, 10000, 500),  # Mid-life car
    'Battery_Voltage': np.random.normal(14.2, 0.2, 500), # Healthy alternator
    'Outside_Temp': np.random.normal(25, 5, 500),     # Nice weather (25C)
    'Status': ['Normal'] * 500
})

# 2. Generate "Anomaly" data (Engine failing, battery dying, overheating)
anomaly_data = pd.DataFrame({
    'Temperature': np.random.normal(115, 5, 500),     # Overheating!
    'Vibration': np.random.normal(1.5, 0.3, 500),     # Shaking violently!
    'Pressure': np.random.normal(28, 3, 500),         # Low tire pressure
    'Speed': np.random.normal(20, 10, 500),           # Limping along
    'RPM': np.random.normal(5000, 500, 500),          # Engine screaming!
    'Odometer': np.random.normal(120000, 20000, 500), # Older car
    'Battery_Voltage': np.random.normal(11.5, 0.5, 500), # Alternator dying!
    'Outside_Temp': np.random.normal(35, 5, 500),     # Hot summer day
    'Status': ['Anomaly'] * 500
})

# Combine the data
df = pd.concat([normal_data, anomaly_data])

# Split features (X) and target (y)
X = df.drop('Status', axis=1)
y = df['Status']

print("Training Advanced Random Forest Model...")
# We upgrade to a Random Forest because 8 features is more complex!
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save the new brain
joblib.dump(model, 'model.joblib')
print("Model saved successfully as 'model.joblib'!")