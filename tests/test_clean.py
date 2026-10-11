import pandas as pd
from src.data.clean import clean


def test_clean_drops_sentinels():
    df = pd.DataFrame({
        "Unnamed: 0": [1, 2],
        "SeriousDlqin2yrs": [0, 1],
        "RevolvingUtilizationOfUnsecuredLines": [0.5, 3.0],
        "age": [40, 30],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 96],
        "DebtRatio": [0.4, 5.0],
        "MonthlyIncome": [5000.0, 1.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 10],
        "NumberOfTimes90DaysLate": [0, 98],
        "NumberRealEstateLoansOrLines": [0, 1],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 98],
        "NumberOfDependents": [0, 2],
    })
    out = clean(df)
    assert "Unnamed: 0" not in out.columns
    assert out["RevolvingUtilizationOfUnsecuredLines"].max() <= 2.0
    assert out["DebtRatio"].max() <= 2.0
    assert out["NumberOfTime30-59DaysPastDueNotWorse"].isna().sum() == 1
    assert out["MonthlyIncome"].isna().sum() == 1
