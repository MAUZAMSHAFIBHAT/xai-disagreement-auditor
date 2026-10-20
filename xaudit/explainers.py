"""Five local feature-attribution methods, implemented from scratch in NumPy.

All methods work on any black-box ``f: (n, d) -> (n,)`` that returns the
probability of the positive class. Every method returns an *attribution*
vector ``phi`` of length ``d`` in probability units, where a positive value
means "this feature pushed the prediction towards the positive class".

Implementing them ourselves (instead of wrapping several libraries) means:

* every method sees the exact same model, background data and baseline, so
  disagreement comes from the methods, not from different preprocessing;
* no fragile compiled dependencies (numba, etc.), so it installs anywhere;
* each estimator is vectorised into a single batched ``f`` call.

References
----------
* Strumbelj & Kononenko (2014), permutation-sampling Shapley values.
* Lundberg & Lee (2017), SHAP.
* Ribeiro, Singh & Guestrin (2016), LIME.
* Sundararajan, Taly & Yan (2017), Integrated Gradients.
* Smilkov et al. (2017), SmoothGrad; Shrikumar et al. (2016), Gradient x Input.
* Zeiler & Fergus (2014), occlusion.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from xaudit.config import IG_STEPS, LIME_SAMPLES, METHODS, SHAPLEY_PERMUTATIONS, SMOOTHGRAD_SAMPLES

PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExplainContext:
    """Everything an explainer needs besides the model and the instance."""

    background: np.ndarray  # (b, d) sample of training rows
    mean: np.ndarray  # (d,) training mean, used as the reference/baseline point
    std: np.ndarray  # (d,) training std (never zero)


def _batched(f: PredictFn, X: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    if len(X) <= chunk:
        return np.asarray(f(X), dtype=float)
    return np.concatenate([np.asarray(f(X[i:i + chunk]), dtype=float) for i in range(0, len(X), chunk)])


# --------------------------------------------------------------------------- #
# 1. Shapley values by permutation sampling (interventional / "KernelSHAP-like")
# --------------------------------------------------------------------------- #

def shapley_sampling(f: PredictFn, x: np.ndarray, ctx: ExplainContext, rng: np.random.Generator,
                     n_permutations: int = SHAPLEY_PERMUTATIONS) -> np.ndarray:
    """Monte-Carlo Shapley values with antithetic permutations.

    For each sampled permutation and background row ``z`` we walk from ``z``
    to ``x`` switching one feature at a time; the change in ``f`` is credited
    to the feature that was switched. Averaging gives an unbiased estimate of
    the interventional Shapley value, and ``sum(phi) ~= f(x) - E[f(z)]``.
    """
    d = x.shape[0]
    half = max(1, n_permutations // 2)
    perms = np.array([rng.permutation(d) for _ in range(half)])
    perms = np.vstack([perms, perms[:, ::-1]])  # antithetic pairs reduce variance
    n = len(perms)
    Z = ctx.background[rng.integers(len(ctx.background), size=n)]

    rank = np.empty_like(perms)
    rank[np.arange(n)[:, None], perms] = np.arange(d)[None, :]
    steps = np.arange(d + 1)
    mask = rank[:, None, :] < steps[None, :, None]  # (n, d+1, d)
    rows = np.where(mask, x[None, None, :], Z[:, None, :])

    preds = _batched(f, rows.reshape(-1, d)).reshape(n, d + 1)
    deltas = np.diff(preds, axis=1)  # deltas[p, j] belongs to feature perms[p, j]
    phi = np.zeros(d)
    np.add.at(phi, perms.ravel(), deltas.ravel())
    return phi / n


# --------------------------------------------------------------------------- #
# 2. LIME - locally weighted linear surrogate
# --------------------------------------------------------------------------- #

def lime(f: PredictFn, x: np.ndarray, ctx: ExplainContext, rng: np.random.Generator,
         n_samples: int = LIME_SAMPLES, kernel_width: float | None = None, ridge: float = 1.0) -> np.ndarray:
    """LIME for tabular data, sampling around the instance in standardised space.

    The surrogate's coefficients are slopes, not attributions. To put LIME on
    the same scale as the additive methods we report
    ``coef_j * (x_j - mean_j) / std_j`` - the contribution of feature ``j``
    relative to the average input under the local linear model.
    """
    d = x.shape[0]
    eps = rng.standard_normal((n_samples, d))
    Z = x[None, :] + eps * ctx.std[None, :]
    Z[0] = x
    eps[0] = 0.0
    y = _batched(f, Z)

    width = kernel_width if kernel_width is not None else 0.75 * np.sqrt(d)
    dist2 = (eps ** 2).sum(axis=1)
    w = np.exp(-dist2 / (width ** 2))
    w = w / w.sum()

    # Weighted ridge regression in closed form (intercept unpenalised).
    A = np.hstack([np.ones((n_samples, 1)), eps])
    W = w[:, None]
    reg = ridge * np.eye(d + 1) / n_samples
    reg[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ (W * A) + reg, A.T @ (w * y))
    coef = beta[1:]
    return coef * (x - ctx.mean) / ctx.std


# --------------------------------------------------------------------------- #
# 3. Integrated Gradients (finite-difference, model-agnostic)
# --------------------------------------------------------------------------- #

def integrated_gradients(f: PredictFn, x: np.ndarray, ctx: ExplainContext, rng: np.random.Generator,
                         n_steps: int = IG_STEPS) -> np.ndarray:
    """Integrated Gradients along the straight path from the mean input to ``x``.

    The path integral of each partial derivative is approximated by summing
    one-feature forward differences whose step equals the path increment.
    For smooth models this converges to standard IG; for tree models (whose
    true gradient is zero almost everywhere) it still captures the jumps the
    path crosses. Completeness holds approximately: ``sum(phi) ~= f(x) - f(mean)``.
    """
    d = x.shape[0]
    base = ctx.mean
    delta = (x - base) / n_steps
    alphas = np.arange(n_steps) / n_steps
    P = base[None, :] + alphas[:, None] * (x - base)[None, :]  # (m, d)
    bumped = np.repeat(P[:, None, :], d, axis=1)  # (m, d, d)
    idx = np.arange(d)
    bumped[:, idx, idx] += delta[None, :]
    preds = _batched(f, np.vstack([P, bumped.reshape(-1, d)]))
    f_P = preds[:n_steps]
    f_B = preds[n_steps:].reshape(n_steps, d)
    return (f_B - f_P[:, None]).sum(axis=0)


# --------------------------------------------------------------------------- #
# 4. SmoothGrad x Input
# --------------------------------------------------------------------------- #

def smoothgrad_x_input(f: PredictFn, x: np.ndarray, ctx: ExplainContext, rng: np.random.Generator,
                       n_samples: int = SMOOTHGRAD_SAMPLES, noise: float = 0.15, h: float = 0.1) -> np.ndarray:
    """Noise-averaged central-difference gradient, times (x - mean).

    A purely *local* method: it only looks at how the model responds to tiny
    nudges around ``x``. It is the method most likely to disagree with the
    others on models with flat regions or sharp thresholds.
    """
    d = x.shape[0]
    X = x[None, :] + rng.standard_normal((n_samples, d)) * noise * ctx.std[None, :]
    step = h * ctx.std
    plus = np.repeat(X[:, None, :], d, axis=1)
    minus = plus.copy()
    idx = np.arange(d)
    plus[:, idx, idx] += step[None, :]
    minus[:, idx, idx] -= step[None, :]
    preds = _batched(f, np.vstack([plus.reshape(-1, d), minus.reshape(-1, d)]))
    fp = preds[: n_samples * d].reshape(n_samples, d)
    fm = preds[n_samples * d:].reshape(n_samples, d)
    grad = ((fp - fm) / (2 * step[None, :])).mean(axis=0)
    return grad * (x - ctx.mean)


# --------------------------------------------------------------------------- #
# 5. Occlusion (marginal feature ablation)
# --------------------------------------------------------------------------- #

def occlusion(f: PredictFn, x: np.ndarray, ctx: ExplainContext, rng: np.random.Generator,
              n_background: int = 40) -> np.ndarray:
    """Drop in prediction when one feature is replaced by values from the data.

    ``phi_j = f(x) - E_z[f(x with x_j := z_j)]``. Ignores interactions, which
    is exactly why it can disagree with Shapley values.
    """
    d = x.shape[0]
    B = ctx.background[rng.choice(len(ctx.background), size=min(n_background, len(ctx.background)),
                                  replace=False)]
    b = len(B)
    rows = np.repeat(x[None, None, :], d, axis=0).repeat(b, axis=1)  # (d, b, d)
    idx = np.arange(d)
    rows[idx, :, idx] = B.T  # feature j of block j gets background values
    preds = _batched(f, np.vstack([x[None, :], rows.reshape(-1, d)]))
    fx = preds[0]
    return fx - preds[1:].reshape(d, b).mean(axis=1)


def random_attribution(d: int, rng: np.random.Generator) -> np.ndarray:
    """Control condition for faithfulness tests."""
    return rng.standard_normal(d)


EXPLAINERS: dict[str, Callable[..., np.ndarray]] = {
    "shapley": shapley_sampling,
    "lime": lime,
    "integrated_gradients": integrated_gradients,
    "smoothgrad_x_input": smoothgrad_x_input,
    "occlusion": occlusion,
}
assert list(EXPLAINERS) == METHODS


def explain_all(f: PredictFn, x: np.ndarray, ctx: ExplainContext, seed: int = 0,
                methods: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run every method on one instance. Returns (attributions, seconds per method)."""
    methods = methods or METHODS
    out: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    x = np.asarray(x, dtype=float)
    for i, m in enumerate(methods):
        rng = np.random.default_rng([seed, i])
        t0 = time.perf_counter()
        phi = EXPLAINERS[m](f, x, ctx, rng)
        timing[m] = time.perf_counter() - t0
        out[m] = np.nan_to_num(np.asarray(phi, dtype=float))
    return out, timing
