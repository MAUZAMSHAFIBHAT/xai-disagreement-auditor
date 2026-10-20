"""REST API for the XAI Disagreement Auditor.

Run:  uvicorn api.main:app --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from xaudit import __version__
from xaudit.audit import audit_dataset, audit_instance
from xaudit.config import DEFAULT_SEED, DEFAULT_TOP_K, METHOD_LABELS, MODEL_LABELS
from xaudit.data import BUILTIN_LABELS, load_dataset
from xaudit.models import TrainedModel, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("xaudit.api")

app = FastAPI(
    title="XAI Disagreement Auditor API",
    version=__version__,
    description="Run five explanation methods on a prediction, measure how much they agree, "
                "and get an Explanation Trust Score.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


@lru_cache(maxsize=32)
def _model(dataset: str, model: str, seed: int) -> TrainedModel:
    log.info("training %s on %s (seed=%s)", model, dataset, seed)
    return train_model(load_dataset(dataset), model, seed=seed)


def _get_model(dataset: str, model: str, seed: int) -> TrainedModel:
    if dataset not in BUILTIN_LABELS:
        raise HTTPException(404, f"Unknown dataset '{dataset}'. Options: {list(BUILTIN_LABELS)}")
    if model not in MODEL_LABELS:
        raise HTTPException(404, f"Unknown model '{model}'. Options: {list(MODEL_LABELS)}")
    return _model(dataset, model, seed)


class Target(BaseModel):
    dataset: str = Field("credit_risk", examples=["credit_risk"])
    model: str = Field("gradient_boosting", examples=["gradient_boosting"])
    k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    seed: int = Field(DEFAULT_SEED, ge=0)


class InstanceRequest(Target):
    index: int = Field(0, ge=0, description="Row of the held-out test set")


class CustomRequest(Target):
    features: dict[str, float] = Field(..., description="Feature values; missing features use the training mean")


class GlobalRequest(Target):
    n: int = Field(30, ge=2, le=300)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/datasets")
def datasets() -> list[dict]:
    out = []
    for key, label in BUILTIN_LABELS.items():
        ds = load_dataset(key)
        out.append({"key": key, "name": label, "rows": len(ds.X), "features": ds.feature_names,
                    "positive_label": ds.positive_label, "has_ground_truth": ds.true_weights is not None})
    return out


@app.get("/models")
def models() -> dict:
    return {"models": MODEL_LABELS, "methods": METHOD_LABELS}


@app.get("/models/{dataset}/{model}")
def model_info(dataset: str, model: str, seed: int = DEFAULT_SEED) -> dict:
    tm = _get_model(dataset, model, seed)
    return {"dataset": dataset, "model": model, "metrics": tm.metrics, "n_test": len(tm.X_test),
            "features": tm.feature_names}


@app.post("/audit/instance")
def audit_one(req: InstanceRequest) -> dict:
    tm = _get_model(req.dataset, req.model, req.seed)
    if req.index >= len(tm.X_test):
        raise HTTPException(422, f"index must be < {len(tm.X_test)}")
    return audit_instance(tm, req.index, k=req.k, seed=req.seed).to_dict()


@app.post("/audit/custom")
def audit_custom(req: CustomRequest) -> dict:
    tm = _get_model(req.dataset, req.model, req.seed)
    unknown = set(req.features) - set(tm.feature_names)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    x = tm.feature_mean.copy()
    for i, f in enumerate(tm.feature_names):
        if f in req.features:
            x[i] = float(req.features[f])
    if not np.all(np.isfinite(x)):
        raise HTTPException(422, "Feature values must be finite numbers")
    return audit_instance(tm, 0, k=req.k, seed=req.seed, x=x).to_dict()


@app.post("/audit/global")
def audit_many(req: GlobalRequest) -> dict:
    tm = _get_model(req.dataset, req.model, req.seed)
    g = audit_dataset(tm, n=req.n, k=req.k, seed=req.seed)
    return {
        "headline": g.headline(),
        "methods": g.methods,
        "mean_agreement_matrix": g.mean_matrix.round(4).tolist(),
        "faithfulness": g.faithfulness.round(4).to_dict(orient="index"),
        "feature_controversy": g.feature_stats.round(4).to_dict(orient="index"),
        "ground_truth": None if g.ground_truth is None else g.ground_truth.round(4).to_dict(orient="index"),
        "per_instance": g.per_instance.round(4).to_dict(orient="records"),
    }
