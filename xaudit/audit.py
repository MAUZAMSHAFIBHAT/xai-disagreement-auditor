"""High-level audit routines used by the dashboard, the API and the CLI."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaudit.agreement import agreement_report, consensus, pair_score, top_k
from xaudit.config import DEFAULT_TOP_K, METHOD_LABELS, METHODS
from xaudit.explainers import ExplainContext, explain_all, random_attribution
from xaudit.faithfulness import evaluate
from xaudit.models import TrainedModel

ABLATION_BACKGROUND = 50


def context_for(tm: TrainedModel) -> ExplainContext:
    return ExplainContext(background=tm.background, mean=tm.feature_mean, std=tm.feature_std)


def ground_truth_scores(tm: TrainedModel, x: np.ndarray, attrs: dict[str, np.ndarray]) -> pd.DataFrame | None:
    """Compare each method with the known data-generating process (synthetic data only)."""
    w_map = tm.dataset.true_weights
    if not w_map:
        return None
    w = np.array([w_map.get(f, 0.0) for f in tm.feature_names])
    informative = set(np.flatnonzero(w != 0).tolist())
    n_inf = len(informative)
    if n_inf == 0:
        return None
    true_contrib = w * (x - tm.feature_mean) / tm.feature_std
    rows = []
    for m, a in attrs.items():
        top = set(top_k(a, n_inf).tolist())
        prec = len(top & informative) / n_inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = spearmanr(np.abs(a), np.abs(true_contrib))[0]
        # direction check on the informative features the method ranks highly
        hits = [j for j in top & informative if abs(true_contrib[j]) > 1e-9]
        sign_ok = np.mean([np.sign(a[j]) == np.sign(true_contrib[j]) for j in hits]) if hits else np.nan
        rows.append({
            "method": m,
            "true_driver_precision": prec,
            "truth_rank_correlation": 0.0 if not np.isfinite(rc) else float(rc),
            "direction_accuracy": float(sign_ok) if np.isfinite(sign_ok) else np.nan,
            "noise_features_in_top": sorted(tm.feature_names[j] for j in top - informative),
        })
    return pd.DataFrame(rows)


@dataclass
class InstanceAudit:
    index: int
    x: np.ndarray
    probability: float
    predicted_label: str
    true_label: str
    attributions: dict[str, np.ndarray]
    timing: dict[str, float]
    report: dict
    consensus: dict[str, np.ndarray]
    faithfulness: pd.DataFrame
    ground_truth: pd.DataFrame | None
    feature_names: list[str]
    summary: list[str] = field(default_factory=list)

    @property
    def trust_score(self) -> float:
        return self.report["trust_score"]

    @property
    def trust_level(self) -> str:
        return self.report["trust_level"]

    def attribution_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.attributions, index=self.feature_names)
        df["consensus"] = self.consensus["mean"]
        df["contested"] = self.consensus["contested"]
        df["value"] = self.x
        return df

    def to_dict(self) -> dict:
        """JSON-friendly representation (used by the API)."""
        return {
            "index": self.index,
            "probability": self.probability,
            "predicted_label": self.predicted_label,
            "true_label": self.true_label,
            "trust_score": round(self.trust_score, 2),
            "trust_level": self.trust_level,
            "features": {f: float(v) for f, v in zip(self.feature_names, self.x)},
            "attributions": {m: dict(zip(self.feature_names, map(float, a))) for m, a in self.attributions.items()},
            "consensus": dict(zip(self.feature_names, map(float, self.consensus["mean"]))),
            "contested_features": [f for f, c in zip(self.feature_names, self.consensus["contested"]) if c],
            "pairwise": [{"method_a": a, "method_b": b, **{k: round(v, 4) for k, v in m.items()},
                          "agreement": round(pair_score(m), 4)}
                         for (a, b), m in self.report["pairs"].items()],
            "faithfulness": self.faithfulness.to_dict(orient="records"),
            "ground_truth": None if self.ground_truth is None else self.ground_truth.to_dict(orient="records"),
            "timing_seconds": {m: round(t, 4) for m, t in self.timing.items()},
            "summary": self.summary,
        }


def _summarise(names: list[str], attrs: dict[str, np.ndarray], cons: dict, report: dict,
               pos_label: str, neg_label: str, k: int) -> list[str]:
    """Plain-English findings for the presenter."""
    lines: list[str] = []
    n = len(attrs)
    tops = {m: int(top_k(a, 1)[0]) for m, a in attrs.items()}
    counts = pd.Series(list(tops.values())).value_counts()
    lead, lead_n = int(counts.index[0]), int(counts.iloc[0])
    if lead_n == n:
        lines.append(f"All {n} methods agree that **{names[lead]}** is the single most important feature.")
    else:
        others = sorted({names[j] for j in tops.values() if j != lead})
        lines.append(f"**{lead_n} of {n}** methods name **{names[lead]}** as the top feature; "
                     f"the others pick {', '.join('**' + o + '**' for o in others)}.")

    conflict = [j for j in np.flatnonzero(cons["direction_conflict"])]
    for j in conflict[:2]:
        up = [METHOD_LABELS[m] for m, a in attrs.items() if a[j] > 0]
        down = [METHOD_LABELS[m] for m, a in attrs.items() if a[j] < 0]
        lines.append(f"Direction conflict on **{names[j]}**: {', '.join(up)} say it pushes towards "
                     f"*{pos_label}*, while {', '.join(down)} say it pushes towards *{neg_label}*.")

    matrix = report["matrix"]
    methods = report["methods"]
    pairs = [(matrix[i, j], methods[i], methods[j]) for i, j in combinations(range(len(methods)), 2)]
    if pairs:
        hi = max(pairs)
        lo = min(pairs)
        lines.append(f"Closest pair: {METHOD_LABELS[hi[1]]} & {METHOD_LABELS[hi[2]]} "
                     f"({hi[0] * 100:.0f}% agreement). Furthest apart: {METHOD_LABELS[lo[1]]} & "
                     f"{METHOD_LABELS[lo[2]]} ({lo[0] * 100:.0f}%).")
    contested = int(cons["contested"].sum())
    unanimous_top = [names[j] for j in np.flatnonzero(cons["topk_votes"] == n)]
    if unanimous_top:
        lines.append(f"Safe to report: every method has {', '.join('**' + u + '**' for u in unanimous_top[:4])} "
                     f"in its top {k}.")
    lines.append(f"{contested} feature(s) are *contested* (in some methods' top {k} but not others', "
                 f"or pushed in opposite directions).")
    return lines


def audit_instance(tm: TrainedModel, index: int, k: int = DEFAULT_TOP_K, seed: int = 0,
                   x: np.ndarray | None = None) -> InstanceAudit:
    """Full audit of one test-set row (or of a custom input ``x``)."""
    custom = x is not None
    x = np.asarray(x if custom else tm.X_test[index], dtype=float)
    f = tm.predict_proba
    ctx = context_for(tm)
    attrs, timing = explain_all(f, x, ctx, seed=seed + int(index))
    report = agreement_report(attrs, k)
    cons = consensus(attrs, k)

    prob = float(f(x[None, :])[0])
    ds = tm.dataset
    abl_bg = tm.background[:ABLATION_BACKGROUND]
    rng = np.random.default_rng(seed + 999)
    scores, _ = evaluate(f, x, {**attrs, "random": random_attribution(len(x), rng)}, k, abl_bg)
    faith = pd.DataFrame([{"method": m, **v} for m, v in scores.items()])

    audit = InstanceAudit(
        index=int(index),
        x=x,
        probability=prob,
        predicted_label=ds.positive_label if prob >= 0.5 else ds.negative_label,
        true_label="(custom input)" if custom else (ds.positive_label if tm.y_test[index] == 1 else ds.negative_label),
        attributions=attrs,
        timing=timing,
        report=report,
        consensus=cons,
        faithfulness=faith,
        ground_truth=ground_truth_scores(tm, x, attrs),
        feature_names=tm.feature_names,
    )
    audit.summary = _summarise(tm.feature_names, attrs, cons, report, ds.positive_label, ds.negative_label, k)
    return audit


@dataclass
class GlobalAudit:
    per_instance: pd.DataFrame  # one row per audited instance
    mean_matrix: np.ndarray  # method x method mean agreement
    feature_stats: pd.DataFrame  # per-feature controversy
    faithfulness: pd.DataFrame  # per-method mean PGI / PGU
    deletion: pd.DataFrame  # mean deletion curves (k x method)
    ground_truth: pd.DataFrame | None
    methods: list[str]
    k: int

    def headline(self) -> dict:
        df = self.per_instance
        return {
            "n_instances": int(len(df)),
            "mean_trust": float(df["trust_score"].mean()),
            "median_trust": float(df["trust_score"].median()),
            "share_low_trust": float((df["trust_level"] == "Low").mean()),
            "share_high_trust": float((df["trust_level"] == "High").mean()),
            "most_contested_feature": str(self.feature_stats.sort_values("contested_rate", ascending=False)
                                          .index[0]),
            "most_faithful_method": str(self.faithfulness.drop(index="random", errors="ignore")
                                        .sort_values("pgi", ascending=False).index[0]),
        }


def audit_dataset(tm: TrainedModel, n: int = 40, k: int = DEFAULT_TOP_K, seed: int = 0,
                  progress: Callable[[float], None] | None = None) -> GlobalAudit:
    """Audit ``n`` test rows (evenly spread across the predicted-probability range)."""
    f = tm.predict_proba
    ctx = context_for(tm)
    probs = f(tm.X_test)
    order = np.argsort(probs)
    n = int(min(n, len(order)))
    idx = order[np.linspace(0, len(order) - 1, n).round().astype(int)]
    names = tm.feature_names
    d = len(names)
    max_k = min(d, 10)
    abl_bg = tm.background[:ABLATION_BACKGROUND]

    rows, mats, contested, votes, spread = [], [], [], [], []
    faith: dict[str, list] = {m: [] for m in [*METHODS, "random"]}
    dels: dict[str, list] = {m: [] for m in [*METHODS, "random"]}
    gts = []
    rng = np.random.default_rng(seed + 12345)

    for t, i in enumerate(idx):
        x = tm.X_test[i]
        attrs, _ = explain_all(f, x, ctx, seed=seed + int(i))
        rep = agreement_report(attrs, k)
        cons = consensus(attrs, k)
        mats.append(rep["matrix"])
        contested.append(cons["contested"])
        votes.append(cons["topk_votes"])
        spread.append(cons["rank_spread"])
        row = {
            "test_index": int(i),
            "probability": float(probs[i]),
            "confidence": float(abs(probs[i] - 0.5) * 2),
            "true_label": int(tm.y_test[i]),
            "correct": bool((probs[i] >= 0.5) == tm.y_test[i]),
            "trust_score": rep["trust_score"],
            "trust_level": rep["trust_level"],
            "n_contested": int(cons["contested"].sum()),
            "top_feature_consensus": names[int(top_k(cons["mean"], 1)[0])],
        }
        row.update({f"agree_{a}__{b}": pair_score(m) for (a, b), m in rep["pairs"].items()})
        rows.append(row)
        scores, curves = evaluate(f, x, {**attrs, "random": random_attribution(d, rng)}, k, abl_bg, max_k)
        for m in scores:
            faith[m].append((scores[m]["pgi"], scores[m]["pgu"]))
            dels[m].append(curves[m])
        gt = ground_truth_scores(tm, x, attrs)
        if gt is not None:
            gts.append(gt.drop(columns="noise_features_in_top"))
        if progress:
            progress((t + 1) / n)

    per = pd.DataFrame(rows)
    C, V, S = np.array(contested), np.array(votes), np.array(spread)
    feature_stats = pd.DataFrame({
        "contested_rate": C.mean(axis=0),
        "mean_topk_votes": V.mean(axis=0),
        "mean_rank_spread": S.mean(axis=0),
    }, index=names)
    faith_df = pd.DataFrame({m: np.array(v).mean(axis=0) for m, v in faith.items()}, index=["pgi", "pgu"]).T
    del_df = pd.DataFrame({m: np.array(v).mean(axis=0) for m, v in dels.items()})
    del_df.index.name = "features_removed"
    gt_df = None
    if gts:
        gt_df = pd.concat(gts).groupby("method").mean(numeric_only=True).reindex(METHODS)

    return GlobalAudit(
        per_instance=per,
        mean_matrix=np.mean(mats, axis=0),
        feature_stats=feature_stats,
        faithfulness=faith_df,
        deletion=del_df,
        ground_truth=gt_df,
        methods=list(METHODS),
        k=k,
    )
