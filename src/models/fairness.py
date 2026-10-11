from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path("data/processed/CreditScoring_features.csv")
TARGET = "SeriousDlqin2yrs"
SEED = 42
THRESHOLD = 0.20


def load():
    df = pd.read_csv(DATA)
    return df.drop(columns=[TARGET]), df[TARGET]


def segment_report(df, group_col, y_col, pred_col, prob_col):
    rows = []
    for g, sub in df.groupby(group_col, dropna=False):
        y = sub[y_col]
        p = sub[pred_col]
        rows.append({
            "segment": g,
            "n": len(sub),
            "default_rate": y.mean(),
            "approval_rate": (p == 0).mean(),
            "mean_pd": sub[prob_col].mean(),
            "auc": roc_auc_score(y, sub[prob_col]) if y.nunique() > 1 else np.nan,
            "pr_auc": average_precision_score(y, sub[prob_col]) if y.nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1,
    ).fit(X_tr, y_tr)

    pd_test = model.predict_proba(X_te)[:, 1]
    pred = (pd_test >= THRESHOLD).astype(int)

    df = X_te.copy()
    df["y"] = y_te.values
    df["pred"] = pred
    df["pd"] = pd_test

    # Age buckets
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 35, 50, 65, 120],
        labels=["<35", "35-50", "50-65", "65+"],
    )

    # Income tier (based on training distribution)
    inc_med = X_tr["MonthlyIncome"].median()
    df["income_tier"] = np.where(
        df["MonthlyIncome"].isna(), "unknown",
        np.where(df["MonthlyIncome"] <= inc_med, "low", "high"),
    )

    print(f"=== Fairness @ threshold={THRESHOLD} ===\n")
    print("--- By age group ---")
    print(segment_report(df, "age_group", "y", "pred", "pd").to_string(index=False))

    print("\n--- By income tier ---")
    print(segment_report(df, "income_tier", "y", "pred", "pd").to_string(index=False))

    # Disparate impact ratio (approval rate of least favored / most favored)
    age_tbl = segment_report(df, "age_group", "y", "pred", "pd")
    print("\n--- Disparate impact (approval rate ratio vs. best group) ---")
    age_tbl["DI_ratio"] = age_tbl["approval_rate"] / age_tbl["approval_rate"].max()
    print(age_tbl[["segment", "approval_rate", "DI_ratio"]].to_string(index=False))
    print("\n(EEOC 80% rule: DI < 0.80 signals potential adverse impact)")
