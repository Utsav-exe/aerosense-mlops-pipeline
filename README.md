# AeroSense.AI: MLOps Predictive Maintenance Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-E91E63?style=flat)
![Render](https://img.shields.io/badge/Cloud-Render-46E3B7?style=flat&logo=render&logoColor=white)

🟢 **Live API Documentation:** [AeroSense Swagger UI](https://aerosense-mlops-pipeline.onrender.com/docs)

---

## 📝 Executive Summary
In fleet management and autonomous vehicles, unexpected hardware failure isn't just expensive—it's dangerous. **AeroSense.AI** is an end-to-end cloud MLOps pipeline and native mobile application designed to predict sensor anomalies in real-time and provide immediate, context-aware maintenance recommendations to drivers and engineers.

Instead of outputting a generic "Check Engine" light, this system evaluates an **8-point telemetry stream**. If an anomaly is detected, it uses **Retrieval-Augmented Generation (RAG)** to query a local vector database of historical maintenance logs, instantly triggering a mobile alert to tell the user why the failure is likely happening and how to fix it.

---

## 🏗️ System Architecture
**The Data Flow:**

1.  **Mobile Telemetry Simulation:** A native Android app injects real-time, physics-based OBD-II sensor data across 8 parameters (Temperature, Vibration, Pressure, Speed, RPM, Odometer, Battery Voltage, Outside Temp) using distinct failure scenarios.
2.  **Cloud API Pipeline:** Data is POSTed continuously to a live FastAPI backend hosted on Render.
3.  **ML Inference:** A trained scikit-learn model evaluates the 8-dimensional telemetry for mechanical failures and anomalies with sub-50ms latency.
4.  **Vector Search (RAG):** If an anomaly is flagged, the system queries a ChromaDB instance containing embedded historical maintenance logs.
5.  **Actionable UI Output:** The API returns the diagnostic status, triggering a persistent, looping audio alarm on the mobile device alongside the top 3 most relevant historical fixes in a responsive modal.

---

## 🛠️ Technology Stack
* **Mobile Frontend:** React Native, Expo (EAS Build), @expo/vector-icons, expo-av (Audio Engine)
* **API & Routing:** FastAPI, Uvicorn, CORS Middleware
* **Cloud Deployment:** Render (PaaS)
* **Machine Learning:** scikit-learn (Isolation Forest/Random Forest), joblib, Pandas, NumPy
* **Vector Database & RAG:** ChromaDB, sentence-transformers

---

## ✨ Core Features
* **Native Android Dashboard:** A high-fidelity mobile interface featuring dark mode, dynamic telemetry gauges, and an interactive ML Scenario Tester.
* **Live Cloud Inference:** RESTful API deployed on Render, completely detached from local compute for true distributed processing.
* **Contextual AI (RAG):** Bridges the gap between predictive ML and actionable engineering via vector embeddings.
* **Critical Audio Alerts:** Built-in mobile audio engine that loops a high-decibel warning siren until the user actively acknowledges and resolves the system fault.
* **Hardware-Free Simulation Phase:** Includes built-in testing protocols (Overheat, Cold Start, Gear Slip, Oil Leak, Battery Drain) to stress-test the cloud AI without needing physical vehicle integration.

---

## 🚀 Quick Start / Deployment

### Part 1: The Mobile App (Frontend)
Experience the system directly on your Android device.

1.  Go to the **Releases** tab on the right side of this repository.
2.  Download the `AeroSense-v1.0.0.apk` file to your Android phone.
3.  Install the application (accept "Install from Unknown Sources" if prompted).
4.  Launch AeroSense, bypass the Bluetooth handshake, and use the **ML Scenario Testing** grid to trigger real-time cloud anomalies.

### Part 2: Local Backend Setup (For Developers)
If you want to train the model or run the API locally:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Utsav-exe/aerosense-mlops-pipeline.git](https://github.com/Utsav-exe/aerosense-mlops-pipeline.git)
    cd aerosense-mlops-pipeline
    ```
2.  **Set up the Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the Model & Generate Vector DB**
    This will generate the synthetic training data, build the `model.joblib`, and index the RAG database.
    ```bash
    python train_model.py
    ```

> **Note:** The mobile app communicates with the live Render cloud by default. You do not need to run the backend locally to use the APK.

---

## 🗺️ Future Scope & Roadmap
To scale this from a Proof-of-Concept to an Enterprise-grade system, the following features are in active development:

* [x] **Native Mobile Frontend:** (Completed in v1.0.0)
* [ ] **Live OBD-II Bluetooth Integration:** Transitioning from the simulated testing environment to bridging the mobile app directly with ELM327 hardware scanners.
* [ ] **Data Drift Detection:** Implement EventuallyAI to monitor incoming sensor distributions and trigger alerts if the ML model's operating environment changes.
* [ ] **Model Registry:** Move model artifacts from local `.joblib` files to an MLflow tracking server.

---

## 👥 The Engineering Team
* **Teammate 1:** Utsav Saxena
* **Teammate 2:** Sarthak Sahu
* **Teammate 3:** Animesh Jha
