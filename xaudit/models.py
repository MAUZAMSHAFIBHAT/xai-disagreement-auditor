"""Model training and a thin wrapper that exposes ``predict_proba`` for the positive class."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xaudit.config import BACKGROUND_SIZE, MODEL_LABELS
from xaudit.data import Dataset


def _build(model_key: str, seed: int):
    if model_key == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=1, random_state=seed)
    if model_key == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                              random_state=seed)
    if model_key == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800,
                          early_stopping=True, random_state=seed),
        )
    raise KeyError(f"Unknown model '{model_key}'. Options: {sorted(MODEL_LABELS)}")


@dataclass
class TrainedModel:
    """A fitted model plus everything the explainers need."""

    model_key: str
    dataset: Dataset
    estimator: object
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    background: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.model_key]

    @property
    def feature_names(self) -> list[str]:
        return self.dataset.feature_names

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, shape (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.estimator.predict_proba(X)[:, 1]


def train_model(dataset: Dataset, model_key: str, seed: int = 42, test_size: float = 0.25) -> TrainedModel:
    X = dataset.X.values.astype(float)
    y = np.asarray(dataset.y).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    t0 = time.perf_counter()
    est = _build(model_key, seed)
    est.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    p = est.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "f1": float(f1_score(y_te, pred)),
        "fit_seconds": float(fit_s),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_tr), size=min(BACKGROUND_SIZE, len(X_tr)), replace=False)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0

    return TrainedModel(
        model_key=model_key,
        dataset=dataset,
        estimator=est,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        background=X_tr[bg_idx],
        feature_mean=X_tr.mean(axis=0),
        feature_std=std,
        metrics=metrics,
    )
