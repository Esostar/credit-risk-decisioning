from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
)

DATA = Path("data/processed/CreditScoring_features.csv")
MODEL_DIR = Path("models")
TARGET = "SeriousDlqin2yrs"
SEED = 42

PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
)
COST_FN = 5
COST_FP = 1
THRESHOLDS = np.arange(0.05, 0.55, 0.05)


def build():
    return lgb.LGBMClassifier(
        **PARAMS, random_state=SEED, n_jobs=-1, verbose=-1
    )


def pick_threshold(y_true, p):
    best_t, best_cost = 0.5, float("inf")
    for t in THRESHOLDS:
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn * COST_FN + fp * COST_FP
        if cost < best_cost:
            best_t, best_cost = t, cost
    return best_t, best_cost


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk")

    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    with mlflow.start_run(run_name="lgbm-calibrated") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_param("calibration", "isotonic")
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_param("seed", SEED)

        cal = CalibratedClassifierCV(build(), cv=5, method="isotonic")
        cal.fit(X_tr, y_tr)

        p = cal.predict_proba(X_te)[:, 1]
        best_t, best_cost = pick_threshold(y_te.values, p)
        pred = (p >= best_t).astype(int)
        cm = confusion_matrix(y_te, pred)

        metrics = {
            "roc_auc": roc_auc_score(y_te, p),
            "pr_auc": average_precision_score(y_te, p),
            "brier": brier_score_loss(y_te, p),
            "log_loss": log_loss(y_te, p),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "optimal_threshold": best_t,
            "total_cost": best_cost,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }
        mlflow.log_metrics(metrics)

        for k, v in metrics.items():
            print(f"{k:20s}: {v}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": cal,
            "threshold": float(best_t),
            "features": list(X.columns),
            "target": TARGET,
            "version": "0.2.0",
        }
        model_path = MODEL_DIR / "credit_risk_v0.2.0.joblib"
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_artifact")
        mlflow.sklearn.log_model(cal, "model")
        print(f"\nSaved: {model_path}")
        print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
