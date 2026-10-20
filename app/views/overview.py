"""Landing page: the question, the approach and the current setup."""

from __future__ import annotations

import streamlit as st

import state
import ui
from xaudit.config import METHOD_FAMILY, METHODS


def render() -> None:
    cfg = state.get_config()
    st.markdown("""
<div class='hero'>
  <div class='eyebrow'>AIPI 590 · Project 1 · Explainable ML</div>
  <h1>When explanations disagree,<br>which one do you trust?</h1>
  <p>SHAP, LIME, Integrated Gradients and other explainers often give <b>different answers for the same
  prediction</b>. This auditor runs five of them side by side, measures how much they agree, turns that
  into an <b>Explanation Trust Score</b> for every prediction, and checks the methods against
  faithfulness tests and ground truth.</p>
  <div class='chips'><span class='chip'>5 explanation methods</span><span class='chip'>6 agreement metrics</span>
  <span class='chip'>Per-prediction trust score</span><span class='chip'>Ground-truth benchmark</span>
  <span class='chip'>REST API</span></div>
</div>""", unsafe_allow_html=True)

    ui.section("How it works")
    c = st.columns(4)
    steps = [
        ("Explain", "Run five attribution methods with the <b>same model, background data and baseline</b>, "
                    "so any disagreement comes from the methods themselves.", "1"),
        ("Compare", "Score every pair of methods with six metrics from Krishna et al. (2022): top-k overlap, "
                    "rank, sign and correlation.", "2"),
        ("Score", "Combine them into a 0–100 <b>Explanation Trust Score</b> and flag the <i>contested</i> "
                  "features that methods disagree on.", "3"),
        ("Verify", "Check who's right with removal-based faithfulness tests and with synthetic data whose "
                   "true drivers are known.", "4"),
    ]
    for col, (t, b, s) in zip(c, steps):
        col.markdown(ui.card(t, b, s), unsafe_allow_html=True)

    ui.section("Current setup", "Change the dataset and model in the sidebar. Every page updates.")
    ds = state.get_dataset(cfg)
    tm = state.get_model(cfg)
    m = tm.metrics
    ui.stat_row([
        ("Dataset", state.dataset_label(cfg), f"{len(ds.X):,} rows · {ds.n_features} features", True),
        ("Model", tm.label, f"trained in {m['fit_seconds']:.2f}s", True),
        ("Test accuracy", f"{m['accuracy']:.1%}", f"F1 {m['f1']:.3f}"),
        ("ROC AUC", f"{m['roc_auc']:.3f}", f"{m['n_test']} held-out rows"),
    ])
    st.write("")
    ui.finding(f"**About this dataset.** {ds.description}")
    if ds.true_weights:
        truth = [f for f, w in ds.true_weights.items() if w != 0]
        ui.finding("**Ground truth available.** The real drivers are "
                   + ", ".join(f"**{t}**" for t in truth)
                   + ". Every other feature is noise or a correlated proxy, so we can check which explanation "
                     "method finds the truth.", "good")

    ui.section("The five methods being audited")
    cols = st.columns(5)
    blurbs = {
        "shapley": "Fair credit from cooperative game theory. Averages each feature's marginal effect over "
                   "random orderings.",
        "lime": "Fits a small weighted linear model to perturbed samples around the prediction.",
        "integrated_gradients": "Integrates the model's sensitivity along a path from an average input to this one.",
        "smoothgrad_x_input": "Local slope of the model, averaged over noisy copies of the input, times the input.",
        "occlusion": "How much the prediction drops when one feature is replaced by typical values.",
    }
    for col, mth in zip(cols, METHODS):
        col.markdown(
            f"<div class='card'><div style='width:28px;height:6px;border-radius:3px;background:{ui.method_color(mth)};"
            f"margin-bottom:10px'></div><h4>{ui.method_label(mth)}</h4>"
            f"<p style='font-size:.78rem;color:{ui.MUTED};margin-bottom:6px'>{METHOD_FAMILY[mth]}</p>"
            f"<p>{blurbs[mth]}</p></div>", unsafe_allow_html=True)

    ui.section("Start exploring")
    b = st.columns(4)
    pages = st.session_state["pages"]
    if b[0].button("Audit one prediction", icon=":material/search:", use_container_width=True, type="primary"):
        st.switch_page(pages["instance"])
    if b[1].button("Audit the whole model", icon=":material/public:", use_container_width=True):
        st.switch_page(pages["global"])
    if b[2].button("Which method is right?", icon=":material/verified:", use_container_width=True):
        st.switch_page(pages["faith"])
    if b[3].button("Ground-truth experiment", icon=":material/science:", use_container_width=True):
        st.switch_page(pages["gt"])
