"""Unit tests for explainers, agreement metrics, faithfulness and data handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaudit.agreement import agreement_report, consensus, pairwise_metrics, top_k
from xaudit.audit import audit_dataset, audit_instance
from xaudit.data import DataValidationError, load_dataset, make_credit_risk, prepare_uploaded
from xaudit.explainers import (ExplainContext, explain_all, integrated_gradients, lime, occlusion,
                               shapley_sampling, smoothgrad_x_input)
from xaudit.faithfulness import evaluate, pgi, pgu
from xaudit.models import train_model


# --------------------------------------------------------------------------- #
# A known linear model: f(x) = w . x + b  (identity link keeps the maths exact)
# --------------------------------------------------------------------------- #
@pytest.fixture
def linear():
    rng = np.random.default_rng(0)
    d = 6
    w = np.array([3.0, -2.0, 1.0, 0.0, 0.5, 0.0])
    bg = rng.standard_normal((200, d))
    ctx = ExplainContext(background=bg, mean=bg.mean(axis=0), std=bg.std(axis=0))
    x = np.array([1.5, 1.0, -2.0, 0.3, 1.0, -0.7])

    def f(X):
        return np.atleast_2d(X) @ w + 0.1

    truth = w * (x - bg.mean(axis=0))  # exact interventional Shapley values of a linear model
    return f, x, ctx, w, truth


def test_shapley_matches_closed_form_on_linear_model(linear):
    f, x, ctx, _, truth = linear
    phi = shapley_sampling(f, x, ctx, np.random.default_rng(1), n_permutations=2000)
    np.testing.assert_allclose(phi, truth, atol=0.15)


def test_shapley_efficiency(linear):
    f, x, ctx, _, _ = linear
    phi = shapley_sampling(f, x, ctx, np.random.default_rng(1), n_permutations=400)
    # sum(phi) = f(x) - mean over the sampled background rows; close to f(x) - E[f]
    assert abs(phi.sum() - (f(x[None])[0] - f(ctx.background).mean())) < 0.2


def test_integrated_gradients_exact_on_linear_model(linear):
    f, x, ctx, _, truth = linear
    phi = integrated_gradients(f, x, ctx, np.random.default_rng(0))
    np.testing.assert_allclose(phi, truth, atol=1e-8)
    assert abs(phi.sum() - (f(x[None])[0] - f(ctx.mean[None])[0])) < 1e-8  # completeness


def test_smoothgrad_x_input_exact_on_linear_model(linear):
    f, x, ctx, _, truth = linear
    phi = smoothgrad_x_input(f, x, ctx, np.random.default_rng(0))
    np.testing.assert_allclose(phi, truth, atol=1e-8)


def test_occlusion_on_linear_model(linear):
    f, x, ctx, _, truth = linear
    phi = occlusion(f, x, ctx, np.random.default_rng(0), n_background=200)
    np.testing.assert_allclose(phi, truth, atol=1e-8)


def test_lime_recovers_linear_model(linear):
    f, x, ctx, _, _ = linear
    phi = lime(f, x, ctx, np.random.default_rng(0), n_samples=3000)
    expected = linear[3] * ctx.std * (x - ctx.mean) / ctx.std  # coef in std units x standardised deviation
    np.testing.assert_allclose(phi, expected, atol=0.05)


def test_all_methods_agree_on_linear_model(linear):
    f, x, ctx, _, _ = linear
    attrs, timing = explain_all(f, x, ctx, seed=0)
    rep = agreement_report(attrs, k=3)
    assert rep["trust_score"] > 85
    assert set(timing) == set(attrs)


def test_shapley_matches_shap_library():
    shap = pytest.importorskip("shap")
    from sklearn.linear_model import LogisticRegression

    ds = load_dataset("diabetes")
    X = ds.X.values
    model = LogisticRegression(max_iter=3000).fit((X - X.mean(0)) / X.std(0), ds.y)
    Xs = (X - X.mean(0)) / X.std(0)

    def f(Z):
        return model.decision_function(np.atleast_2d(Z))

    bg = Xs[:100]
    ctx = ExplainContext(bg, bg.mean(0), bg.std(0))
    ref = np.ravel(shap.explainers.Linear((model.coef_[0], model.intercept_[0]), bg).shap_values(Xs[200:201]))
    ours = shapley_sampling(f, Xs[200], ctx, np.random.default_rng(0), n_permutations=1000)
    assert np.corrcoef(ref, ours)[0, 1] > 0.99


# --------------------------------------------------------------------------- #
# Agreement metrics
# --------------------------------------------------------------------------- #
def test_identical_explanations_agree_perfectly():
    a = np.array([0.5, -0.3, 0.1, 0.05])
    m = pairwise_metrics(a, a * 2, k=2)
    assert all(v == pytest.approx(1.0) for v in m.values())


def test_sign_flip_breaks_sign_agreement_only():
    a = np.array([0.5, -0.3, 0.1, 0.05])
    m = pairwise_metrics(a, -a, k=2)
    assert m["feature_agreement"] == 1.0
    assert m["sign_agreement"] == 0.0
    assert m["rank_correlation"] == pytest.approx(1.0)


def test_disjoint_topk():
    a = np.array([1.0, 0.9, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 0.9])
    m = pairwise_metrics(a, b, k=2)
    assert m["feature_agreement"] == 0.0 and m["rank_agreement"] == 0.0


def test_zero_explanation_is_no_agreement():
    m = pairwise_metrics(np.zeros(4), np.ones(4), k=2)
    assert all(v == 0.0 for v in m.values())


def test_top_k_order():
    assert top_k(np.array([0.1, -0.9, 0.5]), 2).tolist() == [1, 2]


def test_consensus_flags_contested_features():
    attrs = {"a": np.array([1.0, 0.5, 0.0, 0.0]), "b": np.array([1.0, 0.0, 0.5, 0.0]),
             "c": np.array([1.0, 0.5, 0.0, 0.0])}
    c = consensus(attrs, k=2)
    assert not c["contested"][0]  # everyone's #1
    assert c["contested"][1] and c["contested"][2]
    assert not c["contested"][3]


def test_direction_conflict():
    attrs = {"a": np.array([1.0, 0.1]), "b": np.array([-1.0, 0.1])}
    c = consensus(attrs, k=1)
    assert c["direction_conflict"][0] and c["contested"][0]


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #
def test_faithfulness_prefers_true_attribution(linear):
    f, x, ctx, _, truth = linear
    good, bad = truth, -np.abs(truth)[::-1]
    assert pgi(f, x, good, 2, ctx.background) > pgi(f, x, bad, 2, ctx.background)
    assert pgu(f, x, good, 2, ctx.background) < pgu(f, x, bad, 2, ctx.background)


def test_batched_evaluate_matches_single(linear):
    f, x, ctx, _, truth = linear
    scores, curves = evaluate(f, x, {"t": truth}, 2, ctx.background[:50], max_k=4)
    assert scores["t"]["pgi"] == pytest.approx(pgi(f, x, truth, 2, ctx.background[:50]))
    assert scores["t"]["pgu"] == pytest.approx(pgu(f, x, truth, 2, ctx.background[:50]))
    assert len(curves["t"]) == 5 and curves["t"][0] == 0.0


# --------------------------------------------------------------------------- #
# Data & end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["credit_risk", "breast_cancer", "diabetes"])
def test_builtin_datasets(key):
    ds = load_dataset(key)
    assert len(ds.X) == len(ds.y) and set(np.unique(ds.y)) == {0, 1}
    assert not ds.X.isna().any().any()


def test_credit_risk_is_reproducible():
    a, b = make_credit_risk(seed=3), make_credit_risk(seed=3)
    pd.testing.assert_frame_equal(a.X, b.X)


def test_prepare_uploaded_handles_mixed_types():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"num": rng.standard_normal(120), "cat": rng.choice(["a", "b", "c"], 120),
                       "id": [f"id{i}" for i in range(120)], "y": rng.choice(["yes", "no", "maybe"], 120)})
    df.loc[3, "num"] = np.nan
    ds = prepare_uploaded(df, "y", "yes")
    assert "id" not in ds.X.columns and "num" in ds.X.columns
    assert any(c.startswith("cat_") for c in ds.X.columns)
    assert ds.negative_label == "Other"
    assert not ds.X.isna().any().any()


def test_prepare_uploaded_rejects_bad_input():
    df = pd.DataFrame({"a": range(10), "y": [0] * 10})
    with pytest.raises(DataValidationError):
        prepare_uploaded(df, "y")


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "gradient_boosting", "mlp"])
def test_end_to_end_instance_audit(model):
    tm = train_model(load_dataset("credit_risk"), model)
    assert tm.metrics["roc_auc"] > 0.8
    a = audit_instance(tm, 0, k=4)
    assert 0 <= a.trust_score <= 100
    assert a.ground_truth is not None and len(a.ground_truth) == 5
    d = a.to_dict()
    assert set(d["attributions"]) == {"shapley", "lime", "integrated_gradients", "smoothgrad_x_input", "occlusion"}
    assert d["summary"]


def test_global_audit_shapes():
    tm = train_model(load_dataset("diabetes"), "logistic_regression")
    g = audit_dataset(tm, n=6, k=3)
    assert len(g.per_instance) == 6
    assert g.mean_matrix.shape == (5, 5)
    assert "random" in g.faithfulness.index
    # every real method should beat random on a linear model
    assert (g.faithfulness.drop(index="random")["pgi"] > g.faithfulness.loc["random", "pgi"]).all()
    h = g.headline()
    assert 0 <= h["mean_trust"] <= 100
