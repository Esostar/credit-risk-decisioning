from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, average_precision_score

DATA = Path("data/processed/CreditScoring_features.csv")
TARGET = "SeriousDlqin2yrs"
SEED = 42


def load():
    df = pd.read_csv(DATA)
    return df.drop(columns=[TARGET]), df[TARGET]


def build_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate(name, model, X, y, cv):
    pr = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(average_precision_score), n_jobs=-1)
    roc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name:22s} | PR-AUC: {pr.mean():.4f} +/- {pr.std():.4f} | ROC-AUC: {roc.mean():.4f} +/- {roc.std():.4f}")


if __name__ == "__main__":
    X, y = load()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Data: X={X.shape}, y-rate={y.mean():.4f}\n")
    evaluate("LightGBM", build_lgbm(), X, y, cv)
