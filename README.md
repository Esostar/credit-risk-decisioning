# Credit Risk Decisioning

**Live demo:** https://credit-risk-decisioning-42aqunkforj5zxucgk6nql.streamlit.app/

An end-to-end credit risk scoring system: data cleaning, feature engineering,
LightGBM modeling with probability calibration, cost-based threshold selection,
SHAP explainability, fairness auditing, a FastAPI scoring service, an
interactive Streamlit dashboard, MLflow tracking, and Docker packaging.

## Problem

Predict the probability that a borrower will become seriously delinquent
(90+ days past due) within 2 years so lenders can make consistent, auditable
approval decisions that minimize expected loss.

Dataset: Give Me Some Credit (150k borrowers, ~6.7% default rate).

## Results

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Baseline (prior) | 0.0670 | 0.5000 | - |
| Logistic Regression | 0.1775 | 0.8538 | - |
| LightGBM + isotonic | 0.4085 | 0.8651 | 0.0487 |

Business-optimal threshold: 0.20 (vs. naive 0.5) -> saves ~2,136 cost units
when false approval is 5x costlier than false rejection.

Fairness: Disparate impact ratio >= 0.85 across age groups.

## Project structure

    config/           Config files
    dashboards/       Streamlit dashboard
    data/raw/         Original data (gitignored)
    data/processed/   Cleaned + features (gitignored)
    models/           Trained model artifacts (gitignored)
    src/api/          FastAPI scoring service
    src/data/         Data loading + cleaning
    src/features/     Feature engineering
    src/models/       Training, calibration, fairness, explainability
    tests/            Pytest suite

## Quickstart

    pip install -r requirements.txt
    python src/data/clean.py
    python src/features/build.py
    python src/models/train_tracked.py
    python -m uvicorn src.api.main:app --reload --port 8000
    python -m streamlit run dashboards/app.py

## Docker

    docker compose build
    docker compose up -d api
    curl http://localhost:8001/health

## API

POST /score returns probability of default, decision
(approve/review/reject), and top SHAP reasons.

## Testing

    python -m pytest tests/ -v

## MLflow

    python -m mlflow ui --backend-store-uri file:./mlruns

## Design decisions

- Target: SeriousDlqin2yrs = 90+ days past due within 2 years.
- Validation: Stratified 5-fold CV + held-out test set.
- Threshold: selected by minimizing FN*5 + FP*1 cost.
- Calibration: isotonic regression.
- Explainability: SHAP TreeExplainer.
- Fairness: disparate impact ratio, per-segment AUC, PR-AUC.
