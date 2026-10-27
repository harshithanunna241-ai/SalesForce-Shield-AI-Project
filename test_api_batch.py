
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "prepared" / "testing_clean.csv"

df = pd.read_csv(DATA_PATH)

# Select 100 random records for testing
sample = df.sample(n=100, random_state=42)

actual_labels = []
predicted_labels = []

for _, record in sample.iterrows():
    actual_labels.append(int(record["label"]))

    features = record.drop(
        labels=["id", "label", "attack_cat"]
    ).to_dict()

    response = requests.post(
        "http://127.0.0.1:8081/predict",
        json=features,
        timeout=30
    )

    response.raise_for_status()
    result = response.json()
    predicted_labels.append(result["prediction"])

print("\nAPI test completed!")
print("Records tested:", len(sample))

print("\nClassification report:")
print(classification_report(
    actual_labels,
    predicted_labels,
    target_names=["Normal", "Attack"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(actual_labels, predicted_labels))
