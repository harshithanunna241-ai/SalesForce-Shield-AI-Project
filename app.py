
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "shieldai_model.joblib"
INCIDENT_LOG = BASE_DIR / "incidents.csv"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Salesforce ShieldAI",
    description="Intelligent Security Incident Detection and Response API",
    version="1.2.0",
)


class NetworkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int


@app.get("/")
def home():
    return {
        "message": "Salesforce ShieldAI API is running",
        "model_loaded": True,
        "version": "1.2.0",
    }


@app.post("/predict")
def predict(record: NetworkRecord):
    try:
        # Prepare the network record
        record_data = record.model_dump()
        features = pd.DataFrame([record_data])

        # Predict normal traffic or an attack
        prediction = int(model.predict(features)[0])

        # Calculate estimated attack probability
        attack_probability = float(
            model.predict_proba(features)[0][1]
        )

        # Determine risk level and recommended action
        if prediction == 0:
            risk_level = "Low"
            recommended_action = (
                "No attack predicted. Continue monitoring."
            )
        elif attack_probability >= 0.85:
            risk_level = "Critical"
            recommended_action = (
                "Prioritize investigation immediately."
            )
        elif attack_probability >= 0.70:
            risk_level = "High"
            recommended_action = (
                "Investigate this network record."
            )
        else:
            risk_level = "Medium"
            recommended_action = (
                "Review and verify the suspicious network activity."
            )

        # Log model-predicted attacks to incidents.csv
        if prediction == 1:
            incident = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classification": "Attack",
                "attack_probability": round(attack_probability, 4),
                "risk_level": risk_level,
                "recommended_action": recommended_action,
                "network_features": json.dumps(record_data),
            }

            file_exists = (
                INCIDENT_LOG.exists()
                and INCIDENT_LOG.stat().st_size > 0
            )

            with open(
                INCIDENT_LOG,
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(incident.keys()),
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(incident)

        # Return prediction, risk assessment, and logging status
        return {
            "prediction": prediction,
            "classification": (
                "Attack" if prediction == 1 else "Normal"
            ),
            "attack_probability": round(attack_probability, 4),
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "incident_logged": prediction == 1,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc
