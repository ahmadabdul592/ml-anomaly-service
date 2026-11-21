import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))

MERCHANT_CATEGORIES = ["grocery", "fuel", "restaurant", "online_retail", "travel",
                       "electronics", "atm_cash", "gambling"]
NUMERIC_FEATURES = ["amount", "hour", "distance_from_home_km", "txn_count_24h",
                    "avg_amount_30d", "account_age_days", "is_foreign"]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
