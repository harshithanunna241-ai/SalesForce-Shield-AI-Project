from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "shieldai_model.joblib"
OUTPUT_PATH = BASE_DIR / "security_alerts.json"

DROP_COLUMNS = ["label", "attack_cat", "id"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def generate_alerts(model, csv_path):
    # Read a small sample to limit memory usage.
    df = pd.read_csv(csv_path, nrows=1000)

    if df.empty:
        print("No records found in the dataset.")
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return []

    # Preserve network details for alert investigation.
    context_columns = [
        col for col in ["id", "attack_cat", "proto", "service", "state"]
        if col in df.columns
    ]
    context = df[context_columns].copy()

    # Remove labels and identifiers from model input.
    X = df.drop(
        columns=[col for col in DROP_COLUMNS if col in df.columns]
    )

    predictions = model.predict(X)

    probabilities = None
    attack_index = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)

        if 1 in classes:
            attack_index = classes.index(1)

    alerts = []

    for i, prediction in enumerate(predictions):
        if int(prediction) != 1:
            continue

        attack_score = None

        if probabilities is not None and attack_index is not None:
            attack_score = float(probabilities[i][attack_index])

        # Prototype severity thresholds; validate before real use.
        if attack_score is None:
            severity = "unrated"
        elif attack_score >= 0.90:
            severity = "high"
        elif attack_score >= 0.70:
            severity = "medium"
        else:
            severity = "low"

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "network_intrusion_suspected",
            "severity": severity,
            "prediction": "attack",
            "message": (
                "The model classified this network record as a "
                "potential attack. Verify before responding."
            ),
        }

        # Add network context.
        for col in context.columns:
            value = context.iloc[i][col]
            alert[col] = (
                value.item() if hasattr(value, "item") else value
            )

        if attack_score is not None:
            alert["model_attack_score"] = round(attack_score, 4)

        # Actual labels are for evaluation only.
        if "label" in df.columns:
            actual_label = int(df.iloc[i]["label"])
            alert["actual_label_for_evaluation"] = (
                "attack" if actual_label == 1 else "normal"
            )
            alert["prediction_correct"] = (
                actual_label == int(prediction)
            )

        alerts.append(alert)

    # Save generated alerts.
    OUTPUT_PATH.write_text(
        json.dumps(alerts, indent=2, default=str),
        encoding="utf-8"
    )

    print(f"Records checked: {len(df)}")
    print(f"Potential attack alerts: {len(alerts)}")
    print(f"Alerts saved to: {OUTPUT_PATH}")

    # Evaluate all predictions, not only alert records.
    if "label" in df.columns:
        actual = df["label"].astype(int)
        predicted = pd.Series(
            predictions, index=df.index
        ).astype(int)

        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())

        total = len(df)
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        print("\nEvaluation results:")
        print(f"True positives (attacks detected): {tp}")
        print(f"True negatives (normal identified): {tn}")
        print(f"False positives (false alarms): {fp}")
        print(f"False negatives (attacks missed): {fn}")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1-score:  {f1:.2%}")
    else:
        print(
            "Evaluation metrics unavailable: "
            "the dataset has no actual label column."
        )

    return alerts


if __name__ == "__main__":
    model = load_model()

    # Held-out test file based on the corrected dataset mapping.
    test_file = (
        BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
    )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_file}"
        )

    generate_alerts(model, test_file)
