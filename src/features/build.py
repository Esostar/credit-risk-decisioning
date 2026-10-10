from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/processed/CreditScoring_clean.csv")
OUT = Path("data/processed/CreditScoring_features.csv")
TARGET = "SeriousDlqin2yrs"

PAST_DUE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Income-related
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"].fillna(0) + 1)
    df["IsIncomeMissing"] = df["MonthlyIncome"].isna().astype(int)

    # Past-due aggregates
    df["TotalPastDue"] = df[PAST_DUE].fillna(0).sum(axis=1)
    df["HasPastDue"] = (df["TotalPastDue"] > 0).astype(int)
    df["SeverePastDue"] = (
        df["NumberOfTimes90DaysLate"].fillna(0) > 0
    ).astype(int)

    # Utilization + debt flags
    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.5).astype(int)
    df["HighDebtRatio"] = (df["DebtRatio"] > 0.5).astype(int)

    # Age buckets
    df["AgeBin"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120], labels=False
    ).astype("Int64")

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    out = build_features(df)
    out.to_csv(OUT, index=False)

    added = [c for c in out.columns if c not in df.columns]
    print(f"Saved: {OUT}")
    print(f"Shape: {out.shape}")
    print(f"Added features ({len(added)}): {added}")
    print(f"\nTarget rate: {out[TARGET].mean():.4f}")
