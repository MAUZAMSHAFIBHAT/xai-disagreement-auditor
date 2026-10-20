"""Who's Right? Faithfulness tests (and ground truth when available) across many predictions."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import state
import ui
from xaudit.config import METHODS


def _gt_bars(gt: pd.DataFrame) -> go.Figure:
    s = gt["true_driver_precision"].sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=[ui.method_label(m) for m in s.index], y=s.values,
        marker=dict(color=[ui.method_color(m) for m in s.index], line=dict(width=0)),
        text=[f"{v:.0%}" for v in s.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.INK_2),
        hovertemplate="%{x}<br>True-driver precision %{y:.1%}<extra></extra>"))
    fig.update_layout(title=dict(text="True-driver precision (higher = better)", font=dict(size=14, color=ui.INK)),
                      height=360, yaxis_tickformat=".0%", yaxis_range=[0, 1.1])
    fig.update_xaxes(showgrid=False, tickfont=dict(color=ui.INK_2))
    return fig


def render() -> None:
    cfg = state.get_config()
    tm = state.get_model(cfg)
    ui.page_title("Who's Right?", "Disagreement tells you something is off. These tests help decide which "
                                  "explanation to believe.")

    g = state.get_global_audit(cfg, compute=False)
    if g is None:
        ui.finding("The faithfulness tests reuse the **Global Audit** results. Run it once and both pages fill in.")
        if st.button("Run global audit", type="primary", icon=":material/play_arrow:"):
            state.get_global_audit(cfg)
            st.rerun()
        return

    fa = g.faithfulness
    real = fa.drop(index="random")
    best_pgi = real["pgi"].idxmax()
    best_pgu = real["pgu"].idxmin()
    lift = real["pgi"] / max(fa.loc["random", "pgi"], 1e-9)
    ui.stat_row([
        ("Most faithful (PGI)", ui.method_label(best_pgi), f"{real.loc[best_pgi, 'pgi']:.3f} mean prediction change",
         True),
        ("Most sufficient (PGU)", ui.method_label(best_pgu), f"{real.loc[best_pgu, 'pgu']:.3f} change when keeping "
                                                             "top-k only", True),
        ("Weakest method", ui.method_label(real["pgi"].idxmin()), f"only {lift.min():.1f}× better than random", True),
        ("Predictions tested", f"{len(g.per_instance)}", f"top-k = {g.k}, {tm.label}"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        ui.show(charts.method_bars(fa["pgi"], "PGI: remove each method's top-k features (higher = better)"))
    with c2:
        ui.show(charts.method_bars(fa["pgu"], "PGU: keep only each method's top-k (lower = better)",
                                   higher_better=False))
    st.caption("Hatched grey bar = random attributions. A method that barely beats random is not telling you much "
               "about the model.")

    ui.section("Deletion curves", "Remove features one at a time in the order each method ranks them. "
                                  "A faithful ranking makes the prediction move fastest.")
    ui.show(charts.deletion_lines(g.deletion))

    if g.ground_truth is not None:
        ui.section("Ground truth check", "This dataset is synthetic, so we know which features really drive the "
                                         "label. Which method recovers them?")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ui.show(_gt_bars(g.ground_truth))
        with c2:
            gt = g.ground_truth.copy()
            gt.index = [ui.method_label(m) for m in gt.index]
            st.dataframe(gt.rename(columns={"true_driver_precision": "True-driver precision",
                                            "truth_rank_correlation": "Rank corr. with truth",
                                            "direction_accuracy": "Direction accuracy"}).round(3),
                         use_container_width=True)
            top = g.ground_truth["true_driver_precision"].idxmax()
            ui.finding(f"**{ui.method_label(top)}** recovers the true drivers most often "
                       f"({g.ground_truth.loc[top, 'true_driver_precision']:.0%} of its top features are real).",
                       "good")

    ui.section("Reading these results honestly")
    ui.finding("**PGI and PGU remove features by swapping in values from the data**, which is the same thing "
               "Occlusion does. That gives Occlusion a home advantage on these tests, so don't read a PGI win for "
               "Occlusion as proof it's the best method. The ground-truth check doesn't have this bias.", "warn")
    ui.finding("**Faithfulness is about the model, ground truth is about the world.** If the model has learned to "
               "use a proxy feature, a faithful explanation *should* highlight the proxy even though it is not a "
               "true driver. When the two tests disagree, that tells you the model relies on shortcuts.")

    ranking = pd.DataFrame({
        "PGI rank": real["pgi"].rank(ascending=False),
        "PGU rank": real["pgu"].rank(ascending=True),
    })
    if g.ground_truth is not None:
        ranking["Ground-truth rank"] = g.ground_truth["true_driver_precision"].reindex(real.index).rank(ascending=False)
    ranking["Average rank"] = ranking.mean(axis=1)
    ranking = ranking.sort_values("Average rank")
    ranking.index = [ui.method_label(m) for m in ranking.index]
    ui.section("Overall scorecard", "Rank 1 = best on that test.")
    st.dataframe(ranking.astype(float).round(1), use_container_width=True)
    _ = METHODS
