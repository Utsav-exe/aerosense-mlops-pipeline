import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import random

# 1. Generate Fake Data
print("Generating fake sensor data...")
data = []
for _ in range(1000):
    temp = random.uniform(70, 90)
    vib = random.uniform(0.1, 0.5)
    press = random.uniform(30, 40)
    label = 0 # 0 means Normal

    if random.random() > 0.9:
        temp = random.uniform(95, 110) # Overheating
        vib = random.uniform(0.6, 1.2) # High vibration
        label = 1 # 1 means Anomaly

    data.append([temp, vib, press, label])

df = pd.DataFrame(data, columns=['Temperature', 'Vibration', 'Pressure', 'Label'])

# 2. Train the Model
print("Training Decision Tree...")
X = df[['Temperature', 'Vibration', 'Pressure']]
y = df['Label']

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# 3. Save the Model
joblib.dump(model, 'model.joblib')
print("Model saved successfully as 'model.joblib'!")