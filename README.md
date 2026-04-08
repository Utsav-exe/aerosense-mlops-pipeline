# AeroSense: MLOps Predictive Maintenance Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%2BDB-FF4F8B)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)

## Executive Summary
In fleet management and autonomous vehicles, unexpected hardware failure isn't just expensive—it's dangerous. **AeroSense** is an end-to-end MLOps pipeline designed to not only predict sensor anomalies in real-time but also provide immediate, context-aware maintenance recommendations to engineers.

Instead of just outputting "Anomaly Detected," this system uses **Retrieval-Augmented Generation (RAG)** to query a vector database of historical maintenance logs, instantly telling the mechanic *why* the failure is likely happening and *how* to fix it.

## System Architecture
![AeroSense Architecture Diagram](architecture.png)

### The Data Flow:
1. **Telemetry Ingestion:** Real-time sensor data (Temperature, Vibration, Pressure) hits the FastAPI endpoint.
2. **Inference:** A lightweight Decision Tree model evaluates the telemetry for anomalies (Sub-50ms latency).
3. **Vector Search (RAG):** If an anomaly is flagged, the system queries a local ChromaDB instance containing embedded historical maintenance logs.
4. **Actionable Output:** The API returns the anomaly status alongside the top 3 most relevant historical fixes.

## Technology Stack
* **API & Routing:** FastAPI, Uvicorn
* **Machine Learning:** `scikit-learn` (Decision Tree Classifier), `joblib`
* **Vector Database & RAG:** ChromaDB, `sentence-transformers`
* **Data Processing:** Pandas, NumPy
* **DevOps:** GitHub Actions (Automated Testing & CI/CD)

## Core Features
* **Real-time Inference:** RESTful API designed for low-latency predictions.
* **Contextual AI (RAG):** Bridges the gap between predictive ML and actionable engineering via vector embeddings.
* **Automated CI/CD:** Any push to the `main` branch triggers an automated build and test sequence via GitHub Actions.
* **Environment Isolation:** Fully reproducible environment using virtual environments and strict dependency tracking.

## Quick Start / Local Deployment

Follow these steps to run the pipeline locally on your machine.

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
```

### 4. Train the Model & Generate Vector DB
```bash
python train_model.py
```

### 5. Start the FastAPI Server
```bash
uvicorn main:app --reload
```

### 6. Test the API
Navigate to http://127.0.0.1:8000/docs in your browser to interact with the Swagger UI.
1. Try sending normal sensor parameters (e.g., Temp: 85, Vib: 0.3).
2. Try sending anomaly parameters (e.g., Temp: 105, Vib: 1.1) to trigger the RAG maintenance suggestions!

## Future Scope & Roadmap
To scale this from a Proof-of-Concept to an Enterprise-grade system, the following MLOps features are planned:
1. [ ] Data Drift Detection: Implement EvidentlyAI to monitor incoming sensor distributions and trigger alerts if the model's environment changes.
2. [ ] Model Registry: Move model artifacts from local .joblib files to an MLflow tracking server.
3. [ ] Containerization: Wrap the application in a Docker container for cloud-agnostic deployment.

## Team
1. Teammate 1 : Utsav Saxena
2. Teammate 2 : Sarthak Sahu
3. Teammate 3 : Animesh Jha

## Backend Deployment Link
https://aerosense-mlops-pipeline-sh1j.onrender.com/docs
