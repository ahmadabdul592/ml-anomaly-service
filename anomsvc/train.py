"""Train, evaluate and save the model artifact:  python -m anomsvc.train"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, MODEL_PATH, NUMERIC_FEATURES
from .data import make_transactions


def build_pipeline(contamination: float, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                          ("scale", StandardScaler())]),
         ["amount", "avg_amount_30d", "distance_from_home_km"]),
        ("num2", StandardScaler(),
         [c for c in NUMERIC_FEATURES if c not in ("amount", "avg_amount_30d", "distance_from_home_km")]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre),
                     ("model", IsolationForest(n_estimators=300, contamination=contamination,
                                               random_state=seed, n_jobs=-1))])


def anomaly_score(pipe: Pipeline, X) -> np.ndarray:
    """Higher = more anomalous."""
    return -pipe.score_samples(X)


def train(out: str = MODEL_PATH, n: int = 20_000, contamination: float = 0.02,
          seed: int = 42) -> dict:
    df = make_transactions(n=n, anomaly_rate=contamination, seed=seed)
    X, y = df[FEATURES], df["is_anomaly"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    pipe = build_pipeline(contamination, seed).fit(X_tr)            # unsupervised: labels unused
    tr_scores = anomaly_score(pipe, X_tr)
    te_scores = anomaly_score(pipe, X_te)
    threshold = float(np.quantile(tr_scores, 1 - contamination))
    flagged = te_scores > threshold

    metadata = {
        "model_version": dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d-%H%M%S"),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "threshold": threshold,
        "medium_threshold": float(np.quantile(tr_scores, 0.95)),
        "contamination": contamination,
        "metrics": {
            "roc_auc": float(roc_auc_score(y_te, te_scores)),
            "pr_auc": float(average_precision_score(y_te, te_scores)),
            "precision": float(y_te[flagged].mean()) if flagged.any() else 0.0,
            "recall": float(flagged[y_te == 1].mean()),
            "alert_rate": float(flagged.mean()),
        },
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "metadata": metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    meta = train(a.out, a.n, a.contamination, a.seed)
    print(json.dumps({"saved": a.out, "version": meta["model_version"], **meta["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
