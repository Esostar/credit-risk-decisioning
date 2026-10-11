from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

DATA = Path("data/processed/CreditScoring_features.csv")
TARGET = "SeriousDlqin2yrs"
SEED = 42
COST_FN = 5   # approving a defaulter (missed default)
COST_FP = 1   # rejecting a good borrower (lost business)


def load():
    df = pd.read_csv(DATA)
    return df.drop(columns=[TARGET]), df[TARGET]


def base_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1,
    )


if __name__ == "__main__":
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    raw = base_lgbm().fit(X_tr, y_tr)
    p_raw = raw.predict_proba(X_te)[:, 1]

    cal = CalibratedClassifierCV(base_lgbm(), cv=5, method="isotonic")
    cal.fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    print("=== Probability quality (raw -> calibrated) ===")
    print(f"Brier:   {brier_score_loss(y_te, p_raw):.4f} -> {brier_score_loss(y_te, p_cal):.4f}")
    print(f"LogLoss: {log_loss(y_te, p_raw):.4f} -> {log_loss(y_te, p_cal):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_te, p_raw):.4f} -> {roc_auc_score(y_te, p_cal):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_te, p_raw):.4f} -> {average_precision_score(y_te, p_cal):.4f}")

    print(f"\n=== Threshold sweep (cost: FN={COST_FN}x, FP={COST_FP}x) ===")
    print(f"{'thr':>5s} | {'prec':>6s} | {'rec':>6s} | {'f1':>6s} | {'FN':>5s} | {'FP':>6s} | {'cost':>8s}")

    best = None
    for thr in np.arange(0.05, 0.55, 0.05):
        pred = (p_cal >= thr).astype(int)
        fn = int(((pred == 0) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        cost = fn * COST_FN + fp * COST_FP
        print(f"{thr:5.2f} | {prec:6.3f} | {rec:6.3f} | {f1:6.3f} | {fn:5d} | {fp:6d} | {cost:8d}")
        if best is None or cost < best[1]:
            best = (thr, cost)

    print(f"\nBest threshold (min cost): {best[0]:.2f} | total cost: {best[1]}")
