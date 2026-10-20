"""Session configuration and cached computations for the dashboard."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from xaudit.audit import GlobalAudit, InstanceAudit, audit_dataset, audit_instance
from xaudit.config import DEFAULT_SEED, DEFAULT_TOP_K, MODEL_LABELS
from xaudit.data import BUILTIN_LABELS, Dataset, load_dataset
from xaudit.models import TrainedModel, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass
class Config:
    dataset_key: str
    model_key: str
    k: int
    seed: int
    n_global: int

    @property
    def model_label(self) -> str:
        return MODEL_LABELS[self.model_key]


def get_config() -> Config:
    return st.session_state.get("cfg") or Config("credit_risk", "gradient_boosting", DEFAULT_TOP_K, DEFAULT_SEED, 40)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def _builtin_dataset(key: str) -> Dataset:
    return load_dataset(key)


def get_dataset(cfg: Config) -> Dataset:
    if cfg.dataset_key.startswith("uploaded:"):
        ds = st.session_state.get("uploaded_ds")
        if ds is None:
            st.warning("Upload a CSV in the sidebar first.")
            st.stop()
        return ds
    return _builtin_dataset(cfg.dataset_key)


@st.cache_resource(show_spinner="Training model…", max_entries=24)
def _train(dataset_key: str, model_key: str, seed: int, _ds: Dataset) -> TrainedModel:
    return train_model(_ds, model_key, seed=seed)


def get_model(cfg: Config) -> TrainedModel:
    return _train(cfg.dataset_key, cfg.model_key, cfg.seed, get_dataset(cfg))


def get_model_for(cfg: Config, model_key: str) -> TrainedModel:
    return _train(cfg.dataset_key, model_key, cfg.seed, get_dataset(cfg))


@st.cache_resource(show_spinner=False, max_entries=256)
def _instance(dataset_key: str, model_key: str, seed: int, k: int, index: int, x_key: tuple | None,
              _tm: TrainedModel) -> InstanceAudit:
    x = None if x_key is None else np.array(x_key, dtype=float)
    return audit_instance(_tm, index, k=k, seed=seed, x=x)


def get_instance_audit(cfg: Config, index: int, x: np.ndarray | None = None) -> InstanceAudit:
    tm = get_model(cfg)
    x_key = None if x is None else tuple(float(v) for v in x)
    return _instance(cfg.dataset_key, cfg.model_key, cfg.seed, cfg.k, int(index), x_key, tm)


@st.cache_resource(show_spinner=False)
def _global_store() -> dict:
    return {}


def get_global_audit(cfg: Config, model_key: str | None = None, n: int | None = None,
                     compute: bool = True) -> GlobalAudit | None:
    """Global audit, cached across reruns and sessions. Shows a progress bar while computing."""
    model_key = model_key or cfg.model_key
    n = n or cfg.n_global
    key = (cfg.dataset_key, model_key, cfg.seed, cfg.k, n)
    store = _global_store()
    if key in store:
        return store[key]
    if not compute:
        return None
    tm = get_model_for(cfg, model_key)
    bar = st.progress(0.0, text=f"Auditing {n} predictions of the {MODEL_LABELS[model_key]} with 5 methods…")
    res = audit_dataset(tm, n=n, k=cfg.k, seed=cfg.seed,
                        progress=lambda p: bar.progress(p, text=f"Auditing predictions… {int(p * 100)}%"))
    bar.empty()
    store[key] = res
    return res


def dataset_label(cfg: Config) -> str:
    if cfg.dataset_key.startswith("uploaded:"):
        ds = st.session_state.get("uploaded_ds")
        return ds.name if ds else "Uploaded dataset"
    return BUILTIN_LABELS[cfg.dataset_key]


def upload_key(raw: bytes, target: str, positive: str) -> str:
    h = hashlib.sha1(raw + target.encode() + positive.encode()).hexdigest()[:10]
    return f"uploaded:{h}"


def load_benchmark_results() -> pd.DataFrame | None:
    p = RESULTS_DIR / "benchmark.csv"
    if p.exists():
        return pd.read_csv(p)
    return None
