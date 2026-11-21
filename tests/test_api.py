import pytest
from fastapi.testclient import TestClient

from anomsvc.api import create_app
from anomsvc.train import train

NORMAL = {"amount": 40.0, "hour": 14, "distance_from_home_km": 3.0, "txn_count_24h": 2,
          "avg_amount_30d": 40.0, "account_age_days": 1500, "is_foreign": 0,
          "merchant_category": "grocery"}
FRAUD = {"amount": 2450.0, "hour": 3, "distance_from_home_km": 820.0, "txn_count_24h": 11,
         "avg_amount_30d": 45.0, "account_age_days": 20, "is_foreign": 1,
         "merchant_category": "gambling"}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "model.joblib"
    meta = train(out=str(p), n=6000)
    assert meta["metrics"]["roc_auc"] > 0.9
    return str(p)


@pytest.fixture()
def client(model_path):
    with TestClient(create_app(model_path)) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["threshold"] > info["medium_threshold"] or info["threshold"] > 0
    assert "roc_auc" in info["metrics"]


def test_normal_vs_fraud(client):
    n = client.post("/predict", json=NORMAL).json()
    f = client.post("/predict", json=FRAUD).json()
    assert f["anomaly_score"] > n["anomaly_score"]
    assert f["is_anomaly"] is True and f["risk_level"] == "high"
    assert n["is_anomaly"] is False


def test_batch(client):
    r = client.post("/predict/batch", json={"transactions": [NORMAL, FRAUD, NORMAL]}).json()
    assert r["count"] == 3 and r["n_anomalies"] == 1
    assert [p["is_anomaly"] for p in r["predictions"]] == [False, True, False]


@pytest.mark.parametrize("patch", [
    {"amount": -5}, {"hour": 24}, {"merchant_category": "casino"}, {"is_foreign": 2},
    {"unexpected_field": 1},
])
def test_validation_rejects_bad_input(client, patch):
    assert client.post("/predict", json={**NORMAL, **patch}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_degraded_when_model_missing(tmp_path):
    with TestClient(create_app(str(tmp_path / "nope.joblib"))) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=NORMAL).status_code == 503
