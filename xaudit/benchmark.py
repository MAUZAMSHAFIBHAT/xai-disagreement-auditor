"""Ground-truth benchmark: how does feature correlation affect disagreement and correctness?

For each correlation level ``rho`` and each model we generate an equicorrelated
Gaussian dataset where only the first few features drive the label, train the
model, explain a sample of test points with all five methods, and record:

* the mean Explanation Trust Score (how much methods agree), and
* each method's *true-driver precision* (share of its top features that are
  genuinely informative).

This answers the question the dashboard alone cannot: when methods disagree,
does disagreement actually signal that explanations are wrong?
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from xaudit.agreement import agreement_report
from xaudit.audit import context_for, ground_truth_scores
from xaudit.config import DEFAULT_TOP_K
from xaudit.data import make_ground_truth_benchmark
from xaudit.explainers import explain_all
from xaudit.models import train_model

DEFAULT_RHOS = (0.0, 0.3, 0.6, 0.8, 0.95)
DEFAULT_MODELS = ("logistic_regression", "random_forest", "gradient_boosting", "mlp")


def run_benchmark(rhos=DEFAULT_RHOS, models=DEFAULT_MODELS, n_samples: int = 1500, n_features: int = 8,
                  n_informative: int = 4, n_instances: int = 20, k: int = DEFAULT_TOP_K, seed: int = 0,
                  progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Returns a long DataFrame: one row per (rho, model, instance, method)."""
    records = []
    total = len(rhos) * len(models)
    step = 0
    for rho in rhos:
        ds = make_ground_truth_benchmark(n_samples, n_features, float(rho), n_informative, seed)
        for model_key in models:
            tm = train_model(ds, model_key, seed=seed)
            ctx = context_for(tm)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(tm.X_test), size=min(n_instances, len(tm.X_test)), replace=False)
            for i in idx:
                x = tm.X_test[i]
                attrs, _ = explain_all(tm.predict_proba, x, ctx, seed=seed + int(i))
                rep = agreement_report(attrs, k)
                gt = ground_truth_scores(tm, x, attrs)
                for _, r in gt.iterrows():
                    records.append({
                        "rho": float(rho),
                        "model": model_key,
                        "instance": int(i),
                        "method": r["method"],
                        "trust_score": rep["trust_score"],
                        "true_driver_precision": r["true_driver_precision"],
                        "truth_rank_correlation": r["truth_rank_correlation"],
                        "model_auc": tm.metrics["roc_auc"],
                    })
            step += 1
            if progress:
                progress(step / total)
    return pd.DataFrame(records)


def summarise_benchmark(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate tables used in the dashboard and in the write-up."""
    trust = (df.drop_duplicates(["rho", "model", "instance"])
               .groupby(["rho", "model"])["trust_score"].mean().unstack("model"))
    precision = df.groupby(["rho", "method"])["true_driver_precision"].mean().unstack("method")
    by_model_method = df.groupby(["model", "method"])["true_driver_precision"].mean().unstack("method")
    inst = df.groupby(["rho", "model", "instance"]).agg(trust=("trust_score", "first"),
                                                         precision=("true_driver_precision", "mean"))
    corr = float(np.corrcoef(inst["trust"], inst["precision"])[0, 1]) if len(inst) > 2 else float("nan")
    return {"trust": trust, "precision": precision, "by_model_method": by_model_method,
            "instance_level": inst.reset_index(), "trust_precision_corr": corr}
