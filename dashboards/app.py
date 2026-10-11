import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

MODEL_PATH = ROOT / "models" / "credit_risk_v0.2.0.joblib"
DATA_PATH = ROOT / "data" / "processed" / "CreditScoring_features.csv"
TARGET = "SeriousDlqin2yrs"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents", "IncomePerPerson", "IsIncomeMissing",
    "TotalPastDue", "HasPastDue", "SeverePastDue", "HighUtilization",
    "HighDebtRatio", "AgeBin",
]
PAST_DUE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

st.set_page_config(page_title="Credit Risk Decisioning", layout="wide")


def engineer_row(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    if pd.notna(df.at[0, "MonthlyIncome"]):
        df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"].fillna(0) + 1)
    else:
        df["IncomePerPerson"] = np.nan
    df["IsIncomeMissing"] = int(pd.isna(df.at[0, "MonthlyIncome"]))
    df["TotalPastDue"] = df[PAST_DUE].fillna(0).sum(axis=1)
    df["HasPastDue"] = (df["TotalPastDue"] > 0).astype(int)
    df["SeverePastDue"] = (df["NumberOfTimes90DaysLate"].fillna(0) > 0).astype(int)
    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.5).astype(int)
    df["HighDebtRatio"] = (df["DebtRatio"] > 0.5).astype(int)
    df["AgeBin"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 120], labels=False)
    return df[FEATURES]


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

df = load_data()
X = df[features]
y = df[TARGET]

st.title("Credit Risk Decisioning")
st.caption(f"Model v{artifact['version']} | Decision threshold: {threshold}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Explainability", "Score Applicant"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Default rate", f"{y.mean():.2%}")
    c3.metric("Features", len(features))
    c4.metric("Threshold", threshold)

    st.subheader("Default rate by age group")
    tmp = df.copy()
    tmp["age_group"] = pd.cut(tmp["age"], [0, 35, 50, 65, 120], labels=["<35", "35-50", "50-65", "65+"])
    agg = tmp.groupby("age_group", observed=True).agg(n=(TARGET, "size"), defaults=(TARGET, "sum"))
    agg["default_rate"] = agg["defaults"] / agg["n"]
    fig = px.bar(agg, x=agg.index, y="default_rate", text_auto=".2%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revolving utilization vs default")
    fig2 = px.histogram(tmp, x="RevolvingUtilizationOfUnsecuredLines", color=TARGET,
                        nbins=50, barmode="overlay", opacity=0.7)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    p = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, p)
    ap = average_precision_score(y_te, p)
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("PR-AUC", f"{ap:.4f}")

    fpr, tpr, _ = roc_curve(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP={ap:.3f}"))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost sweep across thresholds (FN=5x, FP=1x)")
    rows = []
    for t in np.arange(0.05, 0.55, 0.05):
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        rows.append({"threshold": t, "FN": fn, "FP": fp, "cost": fn * 5 + fp})
    cost_df = pd.DataFrame(rows)
    fig = px.line(cost_df, x="threshold", y="cost", markers=True)
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="chosen")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Global feature importance (mean |SHAP|)")
    lgb_model = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(lgb_model)
    sample = X.sample(min(2000, len(X)), random_state=42)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False).head(15)
    fig = px.bar(imp[::-1], orientation="h", labels={"value": "mean |SHAP|", "index": "feature"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-applicant explanation")
    idx = st.selectbox("Pick an applicant (row index)", sample.index.tolist()[:50])
    row = sample.loc[[idx]]
    sv_row = explainer.shap_values(row)
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    contrib = pd.Series(sv_row[0], index=features).sort_values(key=abs, ascending=False).head(10)
    pd_value = float(model.predict_proba(row)[:, 1][0])
    st.metric("Predicted probability of default", f"{pd_value:.3f}")
    fig = px.bar(contrib[::-1], orientation="h",
                 color=contrib[::-1].values,
                 color_continuous_scale=["green", "lightgrey", "red"])
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Enter applicant details")
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            util = st.number_input("Revolving Utilization", 0.0, 2.0, 0.3, 0.05)
            age = st.number_input("Age", 18, 120, 40)
            past_30_59 = st.number_input("Past due 30-59d", 0, 50, 0)
            debt_ratio = st.number_input("Debt Ratio", 0.0, 2.0, 0.4, 0.05)
        with c2:
            income = st.number_input("Monthly Income", 0.0, 100000.0, 5000.0, 100.0)
            open_lines = st.number_input("Open credit lines", 0, 60, 6)
            late_90 = st.number_input("90+ days late", 0, 50, 0)
        with c3:
            re_loans = st.number_input("Real estate loans", 0, 50, 0)
            past_60_89 = st.number_input("Past due 60-89d", 0, 50, 0)
            dependents = st.number_input("Dependents", 0.0, 20.0, 0.0, 1.0)

        submitted = st.form_submit_button("Score")

    if submitted:
        row = {
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income if income > 0 else np.nan,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": re_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": past_60_89,
            "NumberOfDependents": dependents,
        }
        X_new = engineer_row(row)
        pd_val = float(model.predict_proba(X_new)[:, 1][0])

        if pd_val < threshold - 0.05:
            decision = "APPROVE"
        elif pd_val > threshold + 0.05:
            decision = "REJECT"
        else:
            decision = "REVIEW"

        color = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}[decision]
        st.markdown(f"### Decision: :{color}[{decision}]")
        st.metric("Probability of default", f"{pd_val:.3f}")

        sv_row = explainer.shap_values(X_new)
        if isinstance(sv_row, list):
            sv_row = sv_row[1]
        contrib = pd.Series(sv_row[0], index=features).sort_values(key=abs, ascending=False).head(8)
        fig = px.bar(contrib[::-1], orientation="h",
                     color=contrib[::-1].values,
                     color_continuous_scale=["green", "lightgrey", "red"])
        fig.update_layout(title="Why this decision?", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)