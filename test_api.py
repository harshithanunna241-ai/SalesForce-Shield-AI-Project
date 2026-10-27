
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "prepared" / "testing_clean.csv"

# Load one real record from the testing dataset
df = pd.read_csv(DATA_PATH)
record = df.iloc[0]

# Save the actual answer before removing target columns
actual_label = int(record["label"])

# Keep only the 42 input features
features = record.drop(
    labels=["id", "label", "attack_cat"]
).to_dict()

# Send the record to the ShieldAI API
response = requests.post(
    "http://127.0.0.1:8081/predict",
    json=features,
    timeout=30
)

print("Actual label:", actual_label)
print("API status code:", response.status_code)
print("ShieldAI response:", response.json())
