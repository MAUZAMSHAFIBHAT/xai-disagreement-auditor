"""Global Audit: disagreement patterns across many predictions and across models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import state
import ui
from xaudit.config import MODEL_LABELS, TRUST_HIGH, TRUST_LOW


def _trust_histogram(per: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for lo, hi, lvl in [(0, TRUST_LOW, "Low"), (TRUST_LOW, TRUST_HIGH, "Medium"), (TRUST_HIGH, 100, "High")]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=ui.STATUS[lvl], opacity=0.07, line_width=0,
                      annotation_text=f"{ui.STATUS_ICON[lvl]} {lvl}", annotation_position="top left",
                      annotation_font=dict(color=ui.INK_2, size=11))
    fig.add_trace(go.Histogram(x=per["trust_score"], xbins=dict(start=0, end=100, size=5),
                               marker=dict(color=ui.BLUE, line=dict(color="white", width=2)),
                               hovertemplate="Trust %{x}<br>%{y} predictions<extra></extra>"))
    fig.update_layout(height=340, xaxis_title="Explanation Trust Score", yaxis_title="Predictions", bargap=0.05)
    fig.update_xaxes(range=[0, 100])
    return fig


def _trust_vs_confidence(per: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for correct, name, color, symbol in [(True, "Correct prediction", ui.BLUE, "circle"),
                                         (False, "Misclassified", "#eb6834", "x")]:
        d = per[per["correct"] == correct]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["probability"], y=d["trust_score"], mode="markers", name=name,
            marker=dict(size=10, color=color, symbol=symbol, line=dict(color="white", width=1.5)),
            customdata=d[["test_index", "top_feature_consensus"]],
            hovertemplate="Row %{customdata[0]}<br>P(positive) = %{x:.2f}<br>Trust = %{y:.0f}"
                          "<br>Top feature: %{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=ui.AXIS, dash="dot"))
    fig.update_layout(height=340, xaxis_title="Predicted probability of positive class",
                      yaxis_title="Trust Score")
    fig.update_xaxes(range=[-0.02, 1.02])
    return fig


def _feature_controversy(fs: pd.DataFrame, top: int = 12) -> go.Figure:
    d = fs.sort_values("contested_rate", ascending=False).head(top)
    fig = go.Figure(go.Bar(
        x=d["contested_rate"], y=d.index, orientation="h", marker=dict(color="#eb6834", line=dict(width=0)),
        customdata=d[["mean_topk_votes", "mean_rank_spread"]],
        text=[f"{v:.0%}" for v in d["contested_rate"]], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Contested in %{x:.0%} of predictions<br>Avg methods ranking it top-k: "
                      "%{customdata[0]:.1f}<br>Rank spread (std): %{customdata[1]:.1f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.INK_2))
    fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, d["contested_rate"].max() * 1.2 + 0.05)])
    fig.update_layout(height=max(300, 28 * len(d) + 60), xaxis_title="Share of predictions where contested")
    return fig


def _model_comparison(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[MODEL_LABELS[m] for m in df.index], y=df["mean_trust"],
        marker=dict(color=[ui.MODEL_COLORS[m] for m in df.index], line=dict(width=0)),
        error_y=dict(type="data", array=df["sd_trust"], color=ui.AXIS, thickness=1.5, width=6),
        text=[f"{v:.0f}" for v in df["mean_trust"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=16), customdata=df[["share_low", "auc"]],
        hovertemplate="%{x}<br>Mean trust %{y:.1f}<br>Low-trust share %{customdata[0]:.0%}"
                      "<br>AUC %{customdata[1]:.3f}<extra></extra>"))
    fig.add_hline(y=TRUST_HIGH, line=dict(color=ui.STATUS["High"], dash="dot"),
                  annotation_text="High trust", annotation_position="top left", annotation_font=dict(color=ui.INK_2, size=11))
    fig.add_hline(y=TRUST_LOW, line=dict(color=ui.STATUS["Low"], dash="dot"),
                  annotation_text="Low trust", annotation_position="top left", annotation_font=dict(color=ui.INK_2, size=11))
    fig.update_layout(height=380, yaxis_title="Mean Explanation Trust Score", yaxis_range=[0, 105], bargap=0.45)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=ui.INK))
    return fig


def render() -> None:
    cfg = state.get_config()
    tm = state.get_model(cfg)
    ui.page_title("Global Audit", f"How much do the explanation methods agree across {cfg.n_global} predictions of "
                                  f"the {tm.label}?")

    g = state.get_global_audit(cfg, compute=False)
    if g is None:
        ui.finding(f"This runs all five methods on **{cfg.n_global} test predictions**, spread evenly across the "
                   "probability range. It takes about 5–30 seconds depending on the model. You can change the "
                   "number in the sidebar under *Advanced*.")
        if st.button("Run global audit", type="primary", icon=":material/play_arrow:"):
            g = state.get_global_audit(cfg)
            st.rerun()
        return

    h = g.headline()
    per = g.per_instance
    ui.stat_row([
        ("Mean trust score", f"{h['mean_trust']:.0f}", f"median {h['median_trust']:.0f} of 100"),
        ("High-trust predictions", f"{h['share_high_trust']:.0%}", f"score ≥ {TRUST_HIGH:.0f}"),
        ("Low-trust predictions", f"{h['share_low_trust']:.0%}", f"score < {TRUST_LOW:.0f}: don't show one explanation"),
        ("Most contested feature", h["most_contested_feature"], "methods disagree on it most often", True),
    ])

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Distribution of trust", "Each bar counts predictions in a 5-point band of the Trust Score.")
        ui.show(_trust_histogram(per))
    with c2:
        ui.section("Does trust depend on the prediction?",
                   "Do methods disagree more when the model is unsure (near 0.5) or wrong?")
        ui.show(_trust_vs_confidence(per))

    c1, c2 = st.columns([1.1, 1])
    with c1:
        ui.section("Which methods agree with each other?", "Mean pairwise agreement over all audited predictions.")
        ui.show(charts.agreement_matrix(g.mean_matrix, g.methods))
    with c2:
        ui.section("Which features cause disagreement?",
                   f"Share of predictions where the feature is in some methods' top {g.k} but not others', "
                   "or pushed in opposite directions.")
        ui.show(_feature_controversy(g.feature_stats))

    # insight callouts
    m = g.mean_matrix.copy()
    np.fill_diagonal(m, np.nan)
    avg = np.nanmean(m, axis=1)
    loner = g.methods[int(np.nanargmin(avg))]
    ui.finding(f"**{ui.method_label(loner)}** is the odd one out: its average agreement with the other "
               f"methods is only {np.nanmin(avg):.0%}.", "warn")
    conf_corr = per[["confidence", "trust_score"]].corr().iloc[0, 1]
    if np.isfinite(conf_corr):
        ui.finding(f"Correlation between model confidence and trust: **r = {conf_corr:+.2f}**. "
                   + ("Methods agree more when the model is confident." if conf_corr > 0.15 else
                      "Methods disagree more on confident predictions." if conf_corr < -0.15 else
                      "Confidence tells you little about whether explanations agree, so you have to audit."))
    wrong = per[~per["correct"]]
    if len(wrong) >= 2:
        diff = wrong["trust_score"].mean() - per[per["correct"]]["trust_score"].mean()
        cmp = "about the same as correct ones" if abs(diff) < 1 else f"{diff:+.0f} points vs correct ones"
        ui.finding(f"Misclassified predictions have a mean trust of **{wrong['trust_score'].mean():.0f}** "
                   f"({cmp}).", "bad" if diff < -3 else "")

    ui.section("Most contested predictions", "Pick one and open it in the Instance Audit.")
    worst = per.sort_values("trust_score").head(10)[
        ["test_index", "trust_score", "trust_level", "probability", "correct", "n_contested", "top_feature_consensus"]]
    st.dataframe(worst.rename(columns={
        "test_index": "Row", "trust_score": "Trust", "trust_level": "Level", "probability": "P(positive)",
        "correct": "Correct?", "n_contested": "# contested", "top_feature_consensus": "Consensus top feature"}),
        hide_index=True, use_container_width=True,
        column_config={"Trust": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                       "P(positive)": st.column_config.NumberColumn(format="%.2f")})
    c1, c2, _ = st.columns([1, 1, 2])
    pick = c1.selectbox("Row", worst["test_index"].tolist(), label_visibility="collapsed")
    if c2.button("Open in Instance Audit", icon=":material/open_in_new:", use_container_width=True):
        st.session_state["focus_index"] = int(pick)
        st.switch_page(st.session_state["pages"]["instance"])

    # ---- model comparison ----------------------------------------------------
    ui.section("Does the model family matter?",
               "The same audit for all four model types on this dataset. Simpler models often yield explanations "
               "that agree more.")
    n_cmp = min(cfg.n_global, 30)
    have = {mk: state.get_global_audit(cfg, model_key=mk, n=n_cmp, compute=False) for mk in MODEL_LABELS}
    if any(v is None for v in have.values()):
        if st.button(f"Compare all 4 models ({n_cmp} predictions each)", icon=":material/compare_arrows:"):
            for mk in MODEL_LABELS:
                have[mk] = state.get_global_audit(cfg, model_key=mk, n=n_cmp)
            st.rerun()
    else:
        rows = {}
        for mk, ga in have.items():
            rows[mk] = {"mean_trust": ga.per_instance["trust_score"].mean(),
                        "sd_trust": ga.per_instance["trust_score"].std(),
                        "share_low": (ga.per_instance["trust_level"] == "Low").mean(),
                        "auc": state.get_model_for(cfg, mk).metrics["roc_auc"]}
        df = pd.DataFrame(rows).T
        ui.show(_model_comparison(df))
        best = df["mean_trust"].idxmax()
        worst_m = df["mean_trust"].idxmin()
        ui.finding(f"Explanations of the **{MODEL_LABELS[best]}** agree most (mean trust {df.loc[best, 'mean_trust']:.0f}); "
                   f"the **{MODEL_LABELS[worst_m]}** agrees least ({df.loc[worst_m, 'mean_trust']:.0f}). "
                   "Accuracy alone does not tell you how explainable a model is.")

    st.download_button("Download per-prediction results (CSV)", per.to_csv(index=False).encode(),
                       file_name="global_audit.csv", mime="text/csv", icon=":material/download:")
