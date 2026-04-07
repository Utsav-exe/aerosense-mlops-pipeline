import chromadb

# 1. Initialize ChromaDB (This creates a local folder called 'chroma_data' to save the DB)
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Create or load the collection (table) for our logs
collection = chroma_client.get_or_create_collection(name="maintenance_logs")

# 2. Write the 15 Fake Maintenance Logs
logs = [
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
    "Complete system overheating; replaced the central radiator."
]

# 3. Embed and Store the Logs (Only if the database is empty)
if collection.count() == 0:
    print("Initializing Vector DB... Embedding 15 maintenance logs.")
    collection.add(
        documents=logs,
        ids=[f"log_{i}" for i in range(len(logs))],
        metadatas=[{"type": "repair"} for _ in range(len(logs))]
    )
    print("Vector DB successfully populated!")

# 4. The Search Function
def get_maintenance_suggestions(sensor_data_summary: str):
    """
    Takes a description of the anomaly (e.g., "High temp and vibration")
    and returns the top 3 most relevant historical fixes.
    """
    results = collection.query(
        query_texts=[sensor_data_summary],
        n_results=3
    )
    # Return just the list of 3 text documents
    return results['documents'][0]