from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/CreditScoring_features.csv")
MODEL_DIR = Path("models")
TARGET = "SeriousDlqin2yrs"
SEED = 42
THRESHOLD = 0.20


def build():
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1,
    )


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    cal = CalibratedClassifierCV(build(), cv=5, method="isotonic")
    cal.fit(X_tr, y_tr)

    artifact = {
        "model": cal,
        "threshold": THRESHOLD,
        "features": list(X.columns),
        "target": TARGET,
        "version": "0.1.0",
    }
    path = MODEL_DIR / "credit_risk_v0.1.0.joblib"
    joblib.dump(artifact, path)
    print(f"Saved: {path}")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Threshold: {THRESHOLD}")
