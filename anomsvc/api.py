"""FastAPI service.  Run:  uvicorn anomsvc.api:app --port 8000   (docs at /docs)"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .config import FEATURES, MAX_BATCH, MODEL_PATH
from .schemas import BatchRequest, BatchResponse, Prediction, Transaction
from .train import anomaly_score

log = logging.getLogger("anomsvc")


def create_app(model_path: str | None = None) -> FastAPI:
    path = Path(model_path or MODEL_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        if path.exists():
            app.state.bundle = joblib.load(path)
            log.info("Loaded model %s", app.state.bundle["metadata"]["model_version"])
        else:
            log.warning("Model file %s not found; run `python -m anomsvc.train`", path)
        yield

    app = FastAPI(title="Transaction Anomaly Detection API", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return response

    def bundle(request: Request) -> dict:
        b = request.app.state.bundle
        if b is None:
            raise HTTPException(503, "Model not loaded. Train one with `python -m anomsvc.train`.")
        return b

    def predict_rows(b: dict, rows: list[Transaction]) -> list[Prediction]:
        meta = b["metadata"]
        X = pd.DataFrame([r.model_dump() for r in rows])[FEATURES]
        scores = anomaly_score(b["pipeline"], X)
        out = []
        for s in scores:
            s = float(s)
            level = "high" if s > meta["threshold"] else "medium" if s > meta["medium_threshold"] else "low"
            out.append(Prediction(anomaly_score=round(s, 6), is_anomaly=s > meta["threshold"],
                                  risk_level=level))
        return out

    @app.get("/health")
    def health(request: Request):
        loaded = request.app.state.bundle is not None
        return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}

    @app.get("/model-info")
    def model_info(request: Request):
        return bundle(request)["metadata"]

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction, request: Request):
        return predict_rows(bundle(request), [tx])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest, request: Request):
        if len(req.transactions) > MAX_BATCH:
            raise HTTPException(413, f"Batch too large (max {MAX_BATCH}).")
        b = bundle(request)
        preds = predict_rows(b, req.transactions)
        return BatchResponse(model_version=b["metadata"]["model_version"], count=len(preds),
                             n_anomalies=sum(p.is_anomaly for p in preds), predictions=preds)

    return app


app = create_app()
