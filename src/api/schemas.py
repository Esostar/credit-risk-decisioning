from typing import Optional
from pydantic import BaseModel, Field


class Applicant(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: int = Field(..., ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0)
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: Optional[float] = None
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0)
    NumberOfDependents: Optional[float] = None

    model_config = {"populate_by_name": True}


class Decision(BaseModel):
    request_id: str
    timestamp: str
    probability_of_default: float
    decision: str
    threshold: float
    model_version: str
    reasons: list[dict]
