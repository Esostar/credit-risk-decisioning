import numpy as np
import pandas as pd
from src.features.build import build_features


def _sample_df():
    return pd.DataFrame({
        "SeriousDlqin2yrs": [0, 1],
        "RevolvingUtilizationOfUnsecuredLines": [0.3, 1.5],
        "age": [40, 28],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 2],
        "DebtRatio": [0.4, 0.9],
        "MonthlyIncome": [5000.0, np.nan],
        "NumberOfOpenCreditLinesAndLoans": [5, 8],
        "NumberOfTimes90DaysLate": [0, 1],
        "NumberRealEstateLoansOrLines": [0, 1],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
        "NumberOfDependents": [0, 3],
    })


def test_build_features_creates_expected_columns():
    out = build_features(_sample_df())
    for col in ["IncomePerPerson", "IsIncomeMissing", "TotalPastDue",
                "HasPastDue", "SeverePastDue", "HighUtilization",
                "HighDebtRatio", "AgeBin"]:
        assert col in out.columns


def test_aggregates_are_correct():
    out = build_features(_sample_df())
    assert out["TotalPastDue"].iloc[1] == 4
    assert out["HasPastDue"].iloc[1] == 1
    assert out["SeverePastDue"].iloc[1] == 1
    assert out["IsIncomeMissing"].iloc[1] == 1
