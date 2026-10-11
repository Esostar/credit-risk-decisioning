from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from src.api.schemas import Applicant, Decision

MODEL_PATH = Path("models/credit_risk_v0.1.0.joblib")
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
THRESHOLD = ARTIFACT["threshold"]
FEATURES = ARTIFACT["features"]
VERSION = ARTIFACT["version"]
REVIEW_BAND = 0.05  # +/- around threshold -> manual review

app = FastAPI(title="Credit Risk Decisioning API", version=VERSION)


def _engineer(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"].fillna(0) + 1) if pd.notna(df.at[0, "MonthlyIncome"]) else np.nan
    df["IsIncomeMissing"] = int(pd.isna(df.at[0, "MonthlyIncome"]))
    past_due = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
    df["TotalPastDue"] = df[past_due].fillna(0).sum(axis=1)
    df["HasPastDue"] = (df["TotalPastDue"] > 0).astype(int)
    df["SeverePastDue"] = (df["NumberOfTimes90DaysLate"].fillna(0) > 0).astype(int)
    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.5).astype(int)
    df["HighDebtRatio"] = (df["DebtRatio"] > 0.5).astype(int)
    df["AgeBin"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 120], labels=False)
    return df[FEATURES]


def _decide(pd_value: float) -> str:
    if pd_value < THRESHOLD - REVIEW_BAND:
        return "approve"
    if pd_value > THRESHOLD + REVIEW_BAND:
        return "reject"
    return "review"


def _explain(df_row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    lgb_model = MODEL.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(lgb_model)
    sv = explainer.shap_values(df_row)
    if isinstance(sv, list):
        sv = sv[1]
    contrib = pd.Series(sv[0], index=FEATURES).sort_values(key=abs, ascending=False).head(top_k)
    return [{"feature": k, "impact": float(v)} for k, v in contrib.items()]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION}


@app.post("/score", response_model=Decision)
def score(applicant: Applicant):
    try:
        row = applicant.model_dump(by_alias=True)
        X = _engineer(row)
        pd_value = float(MODEL.predict_proba(X)[:, 1][0])
        return Decision(
            probability_of_default=round(pd_value, 4),
            decision=_decide(pd_value),
            threshold=THRESHOLD,
            model_version=VERSION,
            reasons=_explain(X),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
