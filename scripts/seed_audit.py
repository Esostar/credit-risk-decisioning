"""Seed a demo audit DB by scoring a sample of training applicants."""
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "credit_risk_v0.2.0.joblib"
DATA_PATH = ROOT / "data" / "processed" / "CreditScoring_features.csv"
OUT_DB = ROOT / "dashboards" / "assets" / "audit_demo.db"
N_SAMPLES = 120
SEED = 42

SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    probability_of_default REAL NOT NULL,
    decision TEXT NOT NULL,
    threshold REAL NOT NULL,
    features_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scoring_log_ts ON scoring_log(timestamp DESC);
"""


def main():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    features = artifact["features"]
    version = artifact["version"]

    df = pd.read_csv(DATA_PATH)
    sample = df.sample(N_SAMPLES, random_state=SEED).reset_index(drop=True)
    X = sample[features]

    probs = model.predict_proba(X)[:, 1]

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DB.exists():
        OUT_DB.unlink()

    conn = sqlite3.connect(OUT_DB)
    conn.executescript(SCHEMA)

    now = datetime.now(timezone.utc)
    review_band = 0.05

    for i, (_, row) in enumerate(sample.iterrows()):
        p = float(probs[i])
        if p < threshold - review_band:
            decision = "approve"
        elif p > threshold + review_band:
            decision = "reject"
        else:
            decision = "review"

        ts = (now - timedelta(minutes=(N_SAMPLES - i) * 3)).isoformat()
        feat_json = json.dumps(
            {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
            default=str,
        )
        reasons = json.dumps([{"feature": "TotalPastDue", "impact": float(p - 0.2)}])

        conn.execute(
            "INSERT INTO scoring_log "
            "(request_id, timestamp, model_version, probability_of_default, "
            " decision, threshold, features_json, reasons_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ts, version, p, decision, threshold, feat_json, reasons),
        )

    conn.commit()
    conn.close()

    n_approve = sum(1 for p in probs if p < threshold - review_band)
    n_reject = sum(1 for p in probs if p > threshold + review_band)
    n_review = N_SAMPLES - n_approve - n_reject

    print(f"Seeded: {OUT_DB}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Approved: {n_approve} | Rejected: {n_reject} | Review: {n_review}")


if __name__ == "__main__":
    main()
