"""Synthetic card-transaction generator with injected anomalies (for demo + tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MERCHANT_CATEGORIES


def make_transactions(n: int = 20_000, anomaly_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anom = int(n * anomaly_rate)
    n_norm = n - n_anom

    avg30 = rng.lognormal(3.6, 0.5, n_norm)
    normal = pd.DataFrame({
        "amount": avg30 * rng.lognormal(0.0, 0.6, n_norm),
        "hour": np.clip(rng.normal(14, 4.5, n_norm), 0, 23).astype(int),
        "distance_from_home_km": rng.exponential(8, n_norm),
        "txn_count_24h": rng.poisson(2.0, n_norm),
        "avg_amount_30d": avg30,
        "account_age_days": rng.integers(60, 4000, n_norm),
        "is_foreign": (rng.random(n_norm) < 0.03).astype(int),
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES, n_norm, p=[.28, .16, .2, .17, .05, .06, .06, .02]),
    })
    avg30a = rng.lognormal(3.6, 0.5, n_anom)
    anomalies = pd.DataFrame({
        "amount": avg30a * rng.lognormal(1.8, 0.7, n_anom),            # far above usual spend
        "hour": rng.choice([0, 1, 2, 3, 4, 5, 23], n_anom),
        "distance_from_home_km": rng.exponential(350, n_anom),
        "txn_count_24h": rng.poisson(9, n_anom),                       # velocity burst
        "avg_amount_30d": avg30a,
        "account_age_days": rng.integers(1, 90, n_anom),
        "is_foreign": (rng.random(n_anom) < 0.6).astype(int),
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES, n_anom, p=[.02, .02, .03, .3, .1, .25, .13, .15]),
    })
    df = pd.concat([normal.assign(is_anomaly=0), anomalies.assign(is_anomaly=1)], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
