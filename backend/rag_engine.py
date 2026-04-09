import os
import shutil

import chromadb


DB_PATH = "./chroma_data"
COLLECTION_NAME = "maintenance_logs"

LOGS = [
    "Engine temperature exceeded 100 degrees; replaced cooling fan.",
    "High vibration detected in rotor; realigned the main shaft.",
    "Pressure dropped below 30 psi; patched leak in the hydraulic line.",
    "Overheating and high vibration; replaced the worn engine bearing.",
    "Temperature spikes detected; flushed the coolant system.",
    "Sensor reading anomaly; recalibrated the vibration sensor.",
    "Low pressure warning; replaced the primary hydraulic pump.",
    "Engine knocking sound; adjusted the timing belt.",
    "Excessive heat in battery module; replaced thermal paste.",
    "Minor vibration at high speeds; balanced the tires and rotors.",
    "Pressure valve stuck open; cleaned and lubricated the valve.",
    "Random temperature fluctuations; replaced faulty thermostat.",
    "Harsh vibration on startup; tightened engine mounting bolts.",
    "Drop in fluid pressure; replaced the O-ring seals.",
    "Complete system overheating; replaced the central radiator.",
]

_collection = None


def _create_or_load_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("Initializing Vector DB... Embedding 15 maintenance logs.")
        collection.add(
            documents=LOGS,
            ids=[f"log_{i}" for i in range(len(LOGS))],
            metadatas=[{"type": "repair"} for _ in range(len(LOGS))],
        )
        print("Vector DB successfully populated!")

    return collection


def _get_collection():
    global _collection

    if _collection is not None:
        return _collection

    try:
        _collection = _create_or_load_collection()
        return _collection
    except BaseException:
        # Recover from a corrupted/incompatible local Chroma store by rebuilding it.
        if os.path.isdir(DB_PATH):
            shutil.rmtree(DB_PATH, ignore_errors=True)
        _collection = _create_or_load_collection()
        return _collection


def get_maintenance_suggestions(sensor_data_summary: str):
    """
    Takes a description of the anomaly (e.g., "High temp and vibration")
    and returns the top 3 most relevant historical fixes.
    """
    collection = _get_collection()
    results = collection.query(query_texts=[sensor_data_summary], n_results=3)
    return results["documents"][0]