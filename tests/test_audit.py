import os
import tempfile
from pathlib import Path

import pytest

from src.api import audit


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    db = tmp_path / "test_decisions.db"
    monkeypatch.setattr(audit, "DB_PATH", db)
    audit.init_db()
    return db


def test_log_and_fetch(temp_db):
    rid = audit.log_decision(
        features={"age": 40, "MonthlyIncome": 5000},
        probability_of_default=0.35,
        decision="review",
        threshold=0.20,
        model_version="0.2.0",
        reasons=[{"feature": "age", "impact": -0.1}],
    )
    assert rid is not None

    records = audit.fetch_recent(limit=10)
    assert len(records) == 1
    r = records[0]
    assert r["request_id"] == rid
    assert r["decision"] == "review"
    assert r["probability_of_default"] == 0.35
    assert r["features"]["age"] == 40
    assert r["reasons"][0]["feature"] == "age"


def test_fetch_by_id(temp_db):
    rid = audit.log_decision(
        features={"age": 25},
        probability_of_default=0.80,
        decision="reject",
        threshold=0.20,
        model_version="0.2.0",
        reasons=[],
    )
    rec = audit.fetch_by_id(rid)
    assert rec is not None
    assert rec["decision"] == "reject"

    missing = audit.fetch_by_id("nonexistent-id")
    assert missing is None


def test_recent_ordering(temp_db):
    for i in range(3):
        audit.log_decision(
            features={"i": i},
            probability_of_default=0.1 * i,
            decision="approve",
            threshold=0.2,
            model_version="0.2.0",
            reasons=[],
        )
    recs = audit.fetch_recent(limit=10)
    assert len(recs) == 3
    # Most recent first
    ts = [r["timestamp"] for r in recs]
    assert ts == sorted(ts, reverse=True)
