from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_low_risk():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.1,
        "age": 55,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2,
        "MonthlyIncome": 8000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0,
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "review", "reject"}
    assert len(body["reasons"]) == 5


def test_score_high_risk():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 1.5,
        "age": 25,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.9,
        "MonthlyIncome": 2500,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 2,
        "NumberOfDependents": 3,
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    assert r.json()["decision"] == "reject"
