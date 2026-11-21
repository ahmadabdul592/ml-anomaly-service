# ML Anomaly Service: Train, Serve, Containerise

End-to-end ML engineering demo: an **Isolation Forest transaction-anomaly model** trained in a single
scikit-learn `Pipeline`, served by a validated **FastAPI** REST API, packaged with **Docker**, and checked by
**pytest + GitHub Actions CI**.

![CI](https://github.com/ahmadabdul592/ml-anomaly-service/actions/workflows/ci.yml/badge.svg)

```
 data ──► train.py ──► model.joblib (+ metadata.json) ──► FastAPI /predict ──► JSON risk score
          (Pipeline: log/scale + one-hot + IsolationForest)       ▲
                                                        Docker image · CI
```

## Features
- **One artifact, no train/serve skew:** preprocessing and model live in the same pipeline, saved with version, threshold and metrics.
- **Strict input validation** (Pydantic v2): bad ranges, unknown categories and extra fields return `422`.
- **Endpoints:** `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch` (size-limited).
- **Graceful degradation:** with no model file, `/health` reports `degraded` and `/predict` returns `503`.
- **Risk levels:** `high` if score is above the alert threshold, `medium` above the 95th percentile of training scores, else `low`.
- Docker image trains the model during the build, runs as a non-root user and has a health check.

## Run it
```bash
pip install -r requirements-dev.txt
python -m anomsvc.train                  # writes models/model.joblib + model.json
uvicorn anomsvc.api:app --port 8000      # interactive docs: http://localhost:8000/docs
git 
```
Or with Docker: `docker build -t ml-anomaly-service . && docker run --rm -p 8000:8000 ml-anomaly-service`

## Example
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "amount": 2450, "hour": 3, "distance_from_home_km": 820, "txn_count_24h": 11,
  "avg_amount_30d": 45, "account_age_days": 20, "is_foreign": 1, "merchant_category": "electronics"}'
# {"anomaly_score":0.67,"is_anomaly":true,"risk_level":"high"}
```

## Model quality (synthetic data)
Hold-out ROC-AUC 0.999 · PR-AUC 0.97 · precision 0.89 · recall 0.93 at a ~2% alert rate.
These numbers come from synthetic data with injected anomalies, so they demonstrate the pipeline, **not** real-world accuracy.
To use real data, replace `anomsvc/data.py` and the feature lists in `anomsvc/config.py`.

## Production notes / next steps
- Model artifacts are tied to the scikit-learn version, which is why the Docker build trains inside the image.
- Add: request logging/metrics (Prometheus), drift monitoring on score distribution, model registry (MLflow), auth and rate limiting.

MIT licensed.
