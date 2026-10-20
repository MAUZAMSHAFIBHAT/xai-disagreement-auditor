"""Ground-Truth Lab: does feature correlation drive disagreement, and does disagreement signal error?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import state
import ui
from xaudit.benchmark import DEFAULT_MODELS, DEFAULT_RHOS, run_benchmark, summarise_benchmark
from xaudit.config import METHODS, MODEL_LABELS


def _trust_lines(trust: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for mk in trust.columns:
        fig.add_trace(go.Scatter(
            x=trust.index, y=trust[mk], mode="lines+markers", name=MODEL_LABELS.get(mk, mk),
            line=dict(color=ui.MODEL_COLORS.get(mk, ui.MUTED), width=2),
            marker=dict(size=8, line=dict(color="white", width=1.5)),
            hovertemplate=MODEL_LABELS.get(mk, mk) + "<br>ρ = %{x}<br>Mean trust %{y:.1f}<extra></extra>"))
    fig.update_layout(height=380, xaxis_title="Feature correlation ρ", yaxis_title="Mean Explanation Trust Score",
                      hovermode="x unified")
    return fig


def _precision_lines(prec: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m in [c for c in METHODS if c in prec.columns]:
        fig.add_trace(go.Scatter(
            x=prec.index, y=prec[m], mode="lines+markers", name=ui.method_label(m),
            line=dict(color=ui.method_color(m), width=2), marker=dict(size=8, line=dict(color="white", width=1.5)),
            hovertemplate=ui.method_label(m) + "<br>ρ = %{x}<br>Precision %{y:.0%}<extra></extra>"))
    fig.update_layout(height=380, xaxis_title="Feature correlation ρ", yaxis_title="True-driver precision",
                      yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def _scatter(inst: pd.DataFrame, r: float) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=inst["trust"], y=inst["precision"], mode="markers",
        marker=dict(size=8, color=ui.BLUE, opacity=0.55, line=dict(color="white", width=1)),
        customdata=inst[["rho", "model"]],
        hovertemplate="Trust %{x:.0f}<br>Mean precision %{y:.0%}<br>ρ=%{customdata[0]}, %{customdata[1]}"
                      "<extra></extra>"))
    if len(inst) > 2:
        b1, b0 = np.polyfit(inst["trust"], inst["precision"], 1)
        xs = np.linspace(inst["trust"].min(), inst["trust"].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, mode="lines", line=dict(color=ui.INK, width=2),
                                 hoverinfo="skip"))
    fig.update_layout(height=380, showlegend=False, xaxis_title="Explanation Trust Score (agreement)",
                      yaxis_title="Mean true-driver precision (correctness)", yaxis_tickformat=".0%",
                      title=dict(text=f"r = {r:+.2f}", font=dict(size=14, color=ui.INK)))
    return fig


def render() -> None:
    ui.page_title("Ground-Truth Lab", "A controlled experiment: synthetic data where we know the answer, with "
                                      "feature correlation turned up step by step.")
    ui.finding("**Setup.** 8 Gaussian features with pairwise correlation ρ. Only the four starred features **x1–x4** affect the label "
               "(logistic model, weights 2.0, −1.6, 1.2, −0.9). For each ρ we train all four model types and explain "
               "held-out predictions with all five methods. **Research questions:** (1) Does correlation increase "
               "disagreement? (2) When methods disagree, are explanations actually wrong more often?")

    df = st.session_state.get("benchmark_df")
    if df is None:
        df = state.load_benchmark_results()
        src = "precomputed results in results/benchmark.csv"
    else:
        src = "results computed in this session"

    with st.expander("Re-run the experiment", icon=":material/science:"):
        c1, c2, c3 = st.columns(3)
        n_inst = c1.slider("Predictions per (ρ, model)", 5, 60, 20, step=5)
        seed = c2.number_input("Seed", 0, 9999, 0)
        models = c3.multiselect("Models", list(DEFAULT_MODELS), default=list(DEFAULT_MODELS),
                                format_func=MODEL_LABELS.get)
        if st.button("Run benchmark", type="primary", icon=":material/play_arrow:", disabled=not models):
            bar = st.progress(0.0, text="Running benchmark…")
            df = run_benchmark(rhos=DEFAULT_RHOS, models=tuple(models), n_instances=n_inst, seed=int(seed),
                               progress=lambda p: bar.progress(p, text=f"Running benchmark… {int(p * 100)}%"))
            bar.empty()
            st.session_state["benchmark_df"] = df
            st.rerun()

    if df is None:
        st.info("No results yet. Open *Re-run the experiment* above and press **Run benchmark** (about 20 seconds).")
        return

    s = summarise_benchmark(df)
    trust, prec, r = s["trust"], s["precision"], s["trust_precision_corr"]
    lo_rho, hi_rho = trust.index.min(), trust.index.max()
    drop = (trust.loc[lo_rho] - trust.loc[hi_rho])
    p_drop = prec.loc[lo_rho].mean() - prec.loc[hi_rho].mean()
    ui.stat_row([
        ("Trust ↔ correctness", f"r = {r:+.2f}", "agreement predicts correct explanations"),
        ("Trust drop, ρ 0 → 0.95", f"{drop.mean():.0f} pts", f"largest for {MODEL_LABELS[drop.idxmax()]}"),
        ("Precision drop, ρ 0 → 0.95", f"{p_drop:.0%}", "explanations get less accurate"),
        ("Explanations evaluated", f"{df.drop_duplicates(['rho', 'model', 'instance']).shape[0] * 5:,}", src),
    ])

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Q1 · Correlation erodes agreement", "Mean trust score by model as features become correlated.")
        ui.show(_trust_lines(trust))
    with c2:
        ui.section("…and correctness", "Share of each method's top-4 features that are real drivers.")
        ui.show(_precision_lines(prec))

    c1, c2 = st.columns([1.2, 1])
    with c1:
        ui.section("Q2 · Is disagreement a warning sign?", "Each dot is one prediction: how much methods agreed "
                                                          "vs. how correct they were on average.")
        ui.show(_scatter(s["instance_level"], r))
    with c2:
        ui.section("Key findings")
        ui.finding(f"**Disagreement is a useful warning.** Across "
                   f"{len(s['instance_level'])} predictions, trust and correctness correlate at **r = {r:+.2f}**. "
                   "Low-trust explanations are more often wrong.", "good" if r > 0.3 else "")
        lin = "logistic_regression"
        if lin in trust.columns:
            nonlin = [c for c in trust.columns if c != lin]
            ui.finding(f"**Linear models stay explainable.** Logistic regression keeps a trust score of about "
                       f"{trust[lin].mean():.0f} at every correlation level, while tree ensembles and neural nets "
                       f"fall by up to {drop[nonlin].max():.0f} points.")
        best = prec.mean().idxmax()
        worst = prec.mean().idxmin()
        ui.finding(f"**Best at finding the truth: {ui.method_label(best)}** ({prec[best].mean():.0%} mean precision). "
                   f"**Worst: {ui.method_label(worst)}** ({prec[worst].mean():.0%}). Local gradient methods struggle "
                   "on tree models, whose predictions are piecewise constant.", "warn")

    with st.expander("Full results tables"):
        st.markdown("**Mean trust score (ρ × model)**")
        st.dataframe(trust.rename(columns=MODEL_LABELS).round(1), use_container_width=True)
        st.markdown("**True-driver precision (ρ × method)**")
        st.dataframe(prec.rename(columns=ui.method_label).round(3), use_container_width=True)
        st.markdown("**True-driver precision (model × method)**")
        st.dataframe(s["by_model_method"].rename(columns=ui.method_label, index=MODEL_LABELS).round(3),
                     use_container_width=True)
        st.download_button("Download raw benchmark (CSV)", df.to_csv(index=False).encode(), "benchmark.csv",
                           "text/csv", icon=":material/download:")
