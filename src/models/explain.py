from pathlib import Path
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/CreditScoring_features.csv")
OUT_DIR = Path("dashboards/assets")
TARGET = "SeriousDlqin2yrs"
SEED = 42


def load():
    df = pd.read_csv(DATA)
    return df.drop(columns=[TARGET]), df[TARGET]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1,
    ).fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):
        sv = sv[1]

    # Global importance (mean |SHAP|)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X_te.columns).sort_values(ascending=False)
    print("=== Top 10 features (mean |SHAP|) ===")
    print(imp.head(10).to_string())

    # Global summary plot
    plt.figure()
    shap.summary_plot(sv, X_te, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Bar plot (cleaner for dashboards)
    plt.figure()
    shap.summary_plot(sv, X_te, plot_type="bar", show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Single-applicant explanation (highest predicted risk)
    p = model.predict_proba(X_te)[:, 1]
    top_idx = int(np.argmax(p))
    print(f"\n=== Top-risk applicant (idx={top_idx}, PD={p[top_idx]:.3f}, actual={y_te.iloc[top_idx]}) ===")
    contrib = pd.Series(sv[top_idx], index=X_te.columns).sort_values(key=abs, ascending=False)
    print(contrib.head(10).to_string())

    print(f"\nSaved: {OUT_DIR/'shap_summary.png'}")
    print(f"Saved: {OUT_DIR/'shap_bar.png'}")
