import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/audit/decisions.db")

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


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def log_decision(
    features: dict,
    probability_of_default: float,
    decision: str,
    threshold: float,
    model_version: str,
    reasons: list,
) -> str:
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO scoring_log "
            "(request_id, timestamp, model_version, probability_of_default, "
            " decision, threshold, features_json, reasons_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id, ts, model_version, float(probability_of_default),
                decision, float(threshold),
                json.dumps(features, default=str),
                json.dumps(reasons, default=str),
            ),
        )
    return request_id


def fetch_recent(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scoring_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def fetch_by_id(request_id: str) -> Optional[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM scoring_log WHERE request_id = ?", (request_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["features"] = json.loads(d.pop("features_json"))
    d["reasons"] = json.loads(d.pop("reasons_json"))
    return d


init_db()
