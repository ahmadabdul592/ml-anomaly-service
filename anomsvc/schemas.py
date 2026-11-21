from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["grocery", "fuel", "restaurant", "online_retail", "travel",
                   "electronics", "atm_cash", "gambling"]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "amount": 2450.0, "hour": 3, "distance_from_home_km": 820.0, "txn_count_24h": 11,
        "avg_amount_30d": 45.0, "account_age_days": 20, "is_foreign": 1,
        "merchant_category": "electronics"}})

    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home_km: float = Field(ge=0, le=20_000)
    txn_count_24h: int = Field(ge=0, le=1000)
    avg_amount_30d: float = Field(gt=0, le=1_000_000)
    account_age_days: int = Field(ge=0, le=100_000)
    is_foreign: int = Field(ge=0, le=1)
    merchant_category: Category


class Prediction(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    risk_level: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)


class BatchResponse(BaseModel):
    model_version: str
    count: int
    n_anomalies: int
    predictions: list[Prediction]
