from pathlib import Path
from typing import Union
import pandas as pd

SENTINEL = [96, 98]
DROP_COLS = ["Unnamed: 0"]
PAST_DUE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)

    df = df[df["age"] > 0].reset_index(drop=True)

    for c in PAST_DUE:
        df.loc[df[c].isin(SENTINEL), c] = pd.NA

    df.loc[df["MonthlyIncome"].isin([0, 1]), "MonthlyIncome"] = pd.NA

    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=2.0)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=2.0)

    return df


if __name__ == "__main__":
    raw = Path("data/raw/CreditScoring.csv")
    out = Path("data/processed/CreditScoring_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(raw))
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Shape: {df.shape}")
    print(f"Target rate: {df['SeriousDlqin2yrs'].mean():.4f}")
    print(f"Missing:\n{df.isnull().sum()}")