"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_datasets_and_models():
    assert len(client.get("/datasets").json()) == 3
    assert "random_forest" in client.get("/models").json()["models"]


def test_audit_instance():
    r = client.post("/audit/instance", json={"dataset": "diabetes", "model": "logistic_regression", "index": 2})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["trust_score"] <= 100
    assert len(body["pairwise"]) == 10
    assert "X-Process-Time-ms" in r.headers


def test_audit_custom_and_validation():
    r = client.post("/audit/custom", json={"dataset": "credit_risk", "model": "logistic_regression",
                                           "features": {"debt_to_income": 0.6, "num_late_payments": 6}})
    assert r.status_code == 200 and r.json()["probability"] > 0.5
    bad = client.post("/audit/custom", json={"dataset": "credit_risk", "features": {"nope": 1}})
    assert bad.status_code == 422
    assert client.post("/audit/instance", json={"dataset": "nope"}).status_code == 404


def test_audit_global():
    r = client.post("/audit/global", json={"dataset": "diabetes", "model": "logistic_regression", "n": 4})
    assert r.status_code == 200
    assert r.json()["headline"]["n_instances"] == 4
