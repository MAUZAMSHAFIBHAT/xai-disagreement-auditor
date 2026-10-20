"""Faithfulness tests: when methods disagree, which one reflects what the model does?

We use two removal-based tests (Agarwal et al. 2022, OpenXAI):

* **PGI - Prediction Gap on Important features.** Remove the k features a
  method says matter most. A faithful method picks features whose removal
  moves the prediction a lot. Higher is better.
* **PGU - Prediction Gap on Unimportant features.** Remove every feature
  *except* the top-k. A faithful method's top-k alone should preserve the
  prediction. Lower is better.

"Removing" a feature means replacing it with values drawn from the training
data and averaging (marginal ablation), which avoids a single arbitrary
baseline value.
"""

from __future__ import annotations

import numpy as np

from xaudit.agreement import top_k
from xaudit.explainers import PredictFn


def _ablate(f: PredictFn, x: np.ndarray, drop: np.ndarray, background: np.ndarray) -> float:
    """E_z[f(x with features in ``drop`` replaced by z)]."""
    if len(drop) == 0:
        return float(f(x[None, :])[0])
    rows = np.repeat(x[None, :], len(background), axis=0)
    rows[:, drop] = background[:, drop]
    return float(np.mean(f(rows)))


def pgi(f: PredictFn, x: np.ndarray, attr: np.ndarray, k: int, background: np.ndarray) -> float:
    fx = float(f(x[None, :])[0])
    return abs(fx - _ablate(f, x, top_k(attr, k), background))


def pgu(f: PredictFn, x: np.ndarray, attr: np.ndarray, k: int, background: np.ndarray) -> float:
    fx = float(f(x[None, :])[0])
    keep = set(top_k(attr, k).tolist())
    drop = np.array([j for j in range(len(x)) if j not in keep], dtype=int)
    return abs(fx - _ablate(f, x, drop, background))


def deletion_curve(f: PredictFn, x: np.ndarray, attr: np.ndarray, background: np.ndarray,
                   max_k: int | None = None) -> np.ndarray:
    """|f(x) - f(x with top-1..K features removed)| for K = 0..max_k.

    One batched model call for the whole curve.
    """
    d = len(x)
    max_k = min(max_k or d, d)
    order = top_k(attr, max_k)
    b = len(background)
    rows = np.repeat(x[None, None, :], max_k + 1, axis=0).repeat(b, axis=1)  # (K+1, b, d)
    for kk in range(1, max_k + 1):
        cols = order[:kk]
        rows[kk][:, cols] = background[:, cols]
    preds = np.asarray(f(rows.reshape(-1, d))).reshape(max_k + 1, b).mean(axis=1)
    fx = float(f(x[None, :])[0])
    return np.abs(fx - preds)


def evaluate(f: PredictFn, x: np.ndarray, attrs: dict[str, np.ndarray], k: int, background: np.ndarray,
             max_k: int | None = None) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """PGI, PGU and deletion curves for every method in a single batched model call.

    Returns ``({method: {"pgi": .., "pgu": ..}}, {method: deletion_curve})``.
    Deletion curves have length ``max_k + 1`` (0 = nothing removed).
    """
    d = len(x)
    k = max(1, min(k, d))
    max_k = min(max_k or min(d, 10), d)
    b = len(background)
    blocks, layout = [], []
    for m, a in attrs.items():
        order = top_k(a, d)
        drops = [order[:kk] for kk in range(1, max_k + 1)]  # deletion curve
        drops.append(order[:k])  # PGI
        drops.append(order[k:])  # PGU
        for cols in drops:
            rows = np.repeat(x[None, :], b, axis=0)
            if len(cols):
                rows[:, cols] = background[:, cols]
            blocks.append(rows)
        layout.append(m)
    preds = np.asarray(f(np.vstack([x[None, :], *blocks])), dtype=float)
    fx = preds[0]
    means = preds[1:].reshape(-1, b).mean(axis=1).reshape(len(layout), max_k + 2)
    gaps = np.abs(fx - means)
    scores, curves = {}, {}
    for i, m in enumerate(layout):
        curves[m] = np.concatenate([[0.0], gaps[i, :max_k]])
        scores[m] = {"pgi": float(gaps[i, max_k]), "pgu": float(gaps[i, max_k + 1])}
    return scores, curves
