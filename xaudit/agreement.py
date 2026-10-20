"""Agreement metrics between two attribution vectors, and the Explanation Trust Score.

The six pairwise metrics follow Krishna et al. (2022), *The Disagreement
Problem in Explainable Machine Learning: A Practitioner's Perspective*.
All are in [0, 1] except rank correlation, which is in [-1, 1].
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from xaudit.config import TRUST_HIGH, TRUST_LOW

METRICS: list[str] = [
    "feature_agreement",
    "rank_agreement",
    "sign_agreement",
    "signed_rank_agreement",
    "rank_correlation",
    "pairwise_rank_agreement",
]

METRIC_LABELS: dict[str, str] = {
    "feature_agreement": "Feature agreement",
    "rank_agreement": "Rank agreement",
    "sign_agreement": "Sign agreement",
    "signed_rank_agreement": "Signed rank agreement",
    "rank_correlation": "Rank correlation (Spearman)",
    "pairwise_rank_agreement": "Pairwise rank agreement",
}

METRIC_HELP: dict[str, str] = {
    "feature_agreement": "Share of the top-k features that both methods list (order ignored).",
    "rank_agreement": "Share of top-k positions where both methods put the same feature.",
    "sign_agreement": "Share of top-k features shared by both methods AND pushing in the same direction.",
    "signed_rank_agreement": "Same feature, same position, same direction.",
    "rank_correlation": "Spearman correlation of the full importance rankings (-1 to 1).",
    "pairwise_rank_agreement": "For every pair of features, how often both methods agree on which matters more.",
}


def top_k(a: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest |a|, most important first (stable tie-breaking)."""
    return np.argsort(-np.abs(a), kind="stable")[:k]


def _is_degenerate(a: np.ndarray) -> bool:
    return not np.any(np.abs(a) > 1e-12)


def pairwise_metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = len(a)
    k = max(1, min(k, d))
    if _is_degenerate(a) or _is_degenerate(b):
        # An all-zero explanation carries no ranking information: treat as no agreement.
        return {m: 0.0 for m in METRICS}

    ta, tb = top_k(a, k), top_k(b, k)
    shared = set(ta.tolist()) & set(tb.tolist())
    fa = len(shared) / k
    ra = float(np.mean(ta == tb))
    sa = sum(1 for j in shared if np.sign(a[j]) == np.sign(b[j])) / k
    sra = float(np.mean([(ta[i] == tb[i]) and (np.sign(a[ta[i]]) == np.sign(b[tb[i]])) for i in range(k)]))

    if d > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = spearmanr(np.abs(a), np.abs(b))[0]
        rc = 0.0 if not np.isfinite(rc) else float(rc)
        ia, ib = np.abs(a), np.abs(b)
        i, j = np.triu_indices(d, 1)
        pra = float(np.mean(np.sign(ia[i] - ia[j]) == np.sign(ib[i] - ib[j])))
    else:
        rc, pra = 1.0, 1.0

    return {
        "feature_agreement": fa,
        "rank_agreement": ra,
        "sign_agreement": sa,
        "signed_rank_agreement": sra,
        "rank_correlation": rc,
        "pairwise_rank_agreement": pra,
    }


def pair_score(m: dict[str, float]) -> float:
    """Combine one pair's metrics into a single 0-1 agreement number.

    Equal weight on *which* features matter (feature agreement), *which way*
    they push (sign agreement) and the *full ranking* (rank correlation,
    rescaled to 0-1).
    """
    return (m["feature_agreement"] + m["sign_agreement"] + (m["rank_correlation"] + 1) / 2) / 3


def agreement_report(attrs: dict[str, np.ndarray], k: int) -> dict:
    """All pairwise metrics, the agreement matrix and the Explanation Trust Score."""
    methods = list(attrs)
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    n = len(methods)
    matrix = np.eye(n)
    for (i, a), (j, b) in combinations(enumerate(methods), 2):
        m = pairwise_metrics(attrs[a], attrs[b], k)
        pairs[(a, b)] = m
        matrix[i, j] = matrix[j, i] = pair_score(m)

    if pairs:
        mean_metrics = {mt: float(np.mean([p[mt] for p in pairs.values()])) for mt in METRICS}
        trust = 100.0 * float(np.mean([pair_score(p) for p in pairs.values()]))
    else:
        mean_metrics = {mt: 1.0 for mt in METRICS}
        trust = 100.0

    return {
        "methods": methods,
        "pairs": pairs,
        "matrix": matrix,
        "mean_metrics": mean_metrics,
        "trust_score": trust,
        "trust_level": trust_level(trust),
    }


def trust_level(score: float) -> str:
    if score >= TRUST_HIGH:
        return "High"
    if score >= TRUST_LOW:
        return "Medium"
    return "Low"


def normalise(a: np.ndarray) -> np.ndarray:
    """Scale an attribution vector to unit L1 norm (keeps sign)."""
    s = np.abs(a).sum()
    return a / s if s > 0 else a


def consensus(attrs: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Consensus explanation and per-feature disagreement diagnostics.

    Returns arrays of length d:
      * ``mean`` - mean of L1-normalised attributions (the consensus explanation)
      * ``sign_consensus`` - |mean sign| over methods (1 = all agree on direction)
      * ``topk_votes`` - how many methods put the feature in their top-k
      * ``rank_spread`` - std of the feature's rank across methods
      * ``contested`` - bool: some methods rank it top-k while others don't, or
        methods that care about it disagree on its direction
    """
    M = np.vstack([normalise(v) for v in attrs.values()])  # (m, d)
    n_methods, d = M.shape
    k = max(1, min(k, d))
    ranks = np.argsort(np.argsort(-np.abs(M), axis=1, kind="stable"), axis=1)  # 0 = most important
    in_top = ranks < k
    votes = in_top.sum(axis=0)
    signs = np.sign(M)
    sign_cons = np.abs(signs.mean(axis=0))
    # direction conflict among methods that rank it top-k
    conflict = np.zeros(d, dtype=bool)
    for j in range(d):
        s = signs[in_top[:, j], j]
        s = s[s != 0]
        conflict[j] = len(s) > 1 and (s.min() < 0 < s.max())
    contested = ((votes > 0) & (votes < n_methods)) | conflict
    return {
        "mean": M.mean(axis=0),
        "normalised": M,
        "sign_consensus": sign_cons,
        "topk_votes": votes,
        "rank_spread": ranks.std(axis=0),
        "ranks": ranks,
        "direction_conflict": conflict,
        "contested": contested,
    }
