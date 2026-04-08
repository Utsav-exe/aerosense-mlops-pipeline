# AeroSense: MLOps Predictive Maintenance Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%2BDB-FF4F8B)
![Render](https://img.shields.io/badge/Cloud-Render-46E3B7)

**🟢 Live API Documentation:** [AeroSense Swagger UI](https://aerosense-mlops-pipeline-sh1j.onrender.com/docs)

## Executive Summary
In fleet management and autonomous vehicles, unexpected hardware failure isn't just expensive—it's dangerous. **AeroSense** is an end-to-end cloud MLOps pipeline designed to not only predict sensor anomalies in real-time but also provide immediate, context-aware maintenance recommendations to engineers.

Instead of just outputting a generic error, this system evaluates an 8-point telemetry stream. If an anomaly is detected, it uses **Retrieval-Augmented Generation (RAG)** to query a local vector database of historical maintenance logs, instantly telling the mechanic *why* the failure is likely happening and *how* to fix it.

## System Architecture

### The Data Flow:
1. **IoT Telemetry Ingestion:** A local hardware simulator (`car_simulator.py`) generates real-time, physics-based OBD-II sensor data across 8 parameters (Temperature, Vibration, Pressure, Speed, RPM, Odometer, Battery Voltage, Outside Temp).
2. **Cloud API Pipeline:** Data is POSTed every 5 seconds to a live FastAPI backend hosted on Render.
3. **ML Inference:** A trained model evaluates the 8-dimensional telemetry for mechanical failures and anomalies (Sub-50ms latency).
4. **Vector Search (RAG):** If an anomaly is flagged, the system queries a ChromaDB instance containing embedded historical maintenance logs.
5. **Actionable Output:** The API returns the specific status alongside the top 3 most relevant historical fixes.

## Technology Stack
* **API & Routing:** FastAPI, Uvicorn, CORS Middleware
* **Cloud Deployment:** Render (PaaS)
* **Machine Learning:** `scikit-learn` (Random Forest / Decision Tree), `joblib`, Pandas, NumPy
* **Vector Database & RAG:** ChromaDB, `sentence-transformers`
* **IoT Simulation:** Python `requests` module

## Core Features
* **Live Cloud Inference:** RESTful API deployed on Render, completely detached from local compute.
* **Contextual AI (RAG):** Bridges the gap between predictive ML and actionable engineering via vector embeddings.
* **Advanced Telemetry:** Evaluates 8 distinct vehicle data points simultaneously to determine system health.
* **Automated Hardware Simulator:** Includes a built-in Python script that acts as an IoT bridge, injecting real-world driving scenarios (Normal, Overheating, Alternator Failure, Tire Leaks) into the cloud API.
* **Frontend-Ready (CORS):** Backend is pre-configured with open CORS policies, ready to connect to a modern React or Streamlit dashboard.

---

## 🚀 Quick Start / Local Deployment

Follow these steps to run the pipeline and the IoT simulator locally on your machine.

### 1. Clone the repository
```bash
git clone [https://github.com/Utsav-exe/aerosense-mlops-pipeline.git](https://github.com/Utsav-exe/aerosense-mlops-pipeline.git)
cd aerosense-mlops-pipeline
```

### 2. Set up the Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install requests  # Required for the IoT simulator
```
### 4. Train the Model & Generate Vector DB
This will generate the synthetic training data, build the model.joblib, and index the RAG database.
```bash
python train_model.py
```
### 5. Start the Live Telemetry Simulator
You do not need to run the server locally! Our API is already live on Render. You can immediately start sending simulated car data to the cloud by running the IoT Bridge:
```bash
python car_simulator.py
```
Watch your terminal as the car "drives" and interacts with the cloud AI in real-time!

## Future Scope & Roadmap
To scale this from a Proof-of-Concept to an Enterprise-grade system, the following features are planned:

[ ] Frontend Dashboard: A real-time visual UI built in Streamlit/React to display the streaming gauges and AI diagnostics visually.

[ ] Data Drift Detection: Implement EvidentlyAI to monitor incoming sensor distributions and trigger alerts if the model's environment changes.

[ ] Model Registry: Move model artifacts from local .joblib files to an MLflow tracking server.

## The Engineering Team
1. Teammate 1 : Utsav Saxena
2. Teammate 2 : Sarthak Sahu
3. Teammate 3 : Animesh Jha

## Backend Deployment Link
https://aerosense-mlops-pipeline-sh1j.onrender.com/docs
