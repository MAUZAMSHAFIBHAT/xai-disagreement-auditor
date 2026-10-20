"""Plotly figure builders used across pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ui
from xaudit.config import METHOD_SHORT


def _trunc(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def fingerprint(audit, max_features: int = 14, pos: str = "positive", neg: str = "negative") -> go.Figure:
    """Features x methods heatmap of normalised attributions, with each method's rank in the cell."""
    cons = audit.consensus
    M = cons["normalised"]  # (m, d)
    ranks = cons["ranks"]
    methods = list(audit.attributions)
    names = audit.feature_names
    union = set()
    for i in range(len(methods)):
        union |= set(np.argsort(-np.abs(M[i]))[: max(3, min(max_features // 2, len(names)))].tolist())
    feats = sorted(union, key=lambda j: -np.abs(cons["mean"][j]))[:max_features]
    z = M[:, feats].T
    lim = float(np.abs(z).max()) or 1.0
    text = [[f"#{ranks[i, j] + 1}" for i in range(len(methods))] for j in feats]
    ylabels = [("⚠ " if cons["contested"][j] else "") + _trunc(names[j]) for j in feats]
    hover = [[f"<b>{names[j]}</b><br>{ui.method_label(m)}<br>Share of explanation: {M[i, j]:+.1%}"
              f"<br>Rank: #{ranks[i, j] + 1} of {len(names)}" for i, m in enumerate(methods)] for j in feats]
    fig = go.Figure(go.Heatmap(
        z=z, x=[ui.method_label(m) for m in methods], y=ylabels, zmin=-lim, zmax=lim,
        colorscale=ui.DIVERGING, text=text, texttemplate="%{text}", textfont=dict(size=12),
        hovertext=hover, hoverinfo="text", xgap=3, ygap=3,
        colorbar=dict(title=dict(text="Share", side="top", font=dict(size=11)),
                      tickformat="+.0%", thickness=12, len=0.9, outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.INK_2, size=12), showgrid=False)
    fig.update_xaxes(side="top", tickfont=dict(color=ui.INK, size=12), showgrid=False)
    fig.update_layout(height=max(320, 34 * len(feats) + 90), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def side_by_side(audit, max_features: int = 12) -> go.Figure:
    cons = audit.consensus
    M = cons["normalised"]
    methods = list(audit.attributions)
    names = audit.feature_names
    feats = list(np.argsort(-np.abs(cons["mean"]))[:max_features])
    fig = make_subplots(rows=1, cols=len(methods), shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[ui.method_label(m) for m in methods])
    lim = float(np.abs(M[:, feats]).max()) * 1.1 or 1.0
    for i, m in enumerate(methods):
        vals = M[i, feats]
        fig.add_trace(go.Bar(
            x=vals, y=[_trunc(names[j], 22) for j in feats], orientation="h", marker_color=ui.method_color(m),
            marker_line_width=0, name=ui.method_label(m), showlegend=False,
            hovertemplate="<b>%{y}</b><br>" + ui.method_label(m) + ": %{x:+.1%}<extra></extra>",
        ), row=1, col=i + 1)
        fig.update_xaxes(range=[-lim, lim], tickformat=".0%", zeroline=True, zerolinecolor=ui.AXIS,
                         zerolinewidth=1, showgrid=False, row=1, col=i + 1, nticks=3)
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.INK_2))
    fig.update_annotations(font=dict(size=12, color=ui.INK))
    fig.update_layout(height=max(320, 30 * len(feats) + 80), bargap=0.3, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def agreement_matrix(matrix: np.ndarray, methods: list[str], title: str = "") -> go.Figure:
    labels = [ui.method_label(m) for m in methods]
    xlabels = [METHOD_SHORT.get(m, m) for m in methods]
    text = [[f"{v:.0%}" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=xlabels, y=labels, zmin=0, zmax=1, colorscale=[[i / 6, c] for i, c in enumerate(ui.SEQ_BLUE)],
        text=text, texttemplate="%{text}", textfont=dict(size=13), xgap=3, ygap=3,
        customdata=[[labels[j] for j in range(len(labels))] for _ in labels],
        hovertemplate="%{y} vs %{customdata}<br>Agreement: %{z:.0%}<extra></extra>",
        colorbar=dict(title="Agreement", tickformat=".0%", thickness=12, outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=ui.INK))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=ui.INK), tickangle=0)
    fig.update_layout(height=420, title=title)
    return fig


def consensus_bars(audit, max_features: int = 12, pos: str = "positive", neg: str = "negative") -> go.Figure:
    cons = audit.consensus
    names = audit.feature_names
    feats = list(np.argsort(-np.abs(cons["mean"]))[:max_features])
    vals = cons["mean"][feats]
    contested = cons["contested"][feats]
    colors = [ui.RED if v > 0 else ui.BLUE for v in vals]
    patterns = ["/" if c else "" for c in contested]
    votes = cons["topk_votes"][feats]
    n = len(audit.attributions)
    fig = go.Figure(go.Bar(
        x=vals, y=[("⚠ " if c else "") + _trunc(names[j]) for j, c in zip(feats, contested)], orientation="h",
        marker=dict(color=colors, line=dict(width=0), pattern=dict(shape=patterns, fgcolor="white", size=6)),
        customdata=np.stack([votes, cons["sign_consensus"][feats]], axis=1),
        hovertemplate=("<b>%{y}</b><br>Consensus share: %{x:+.1%}<br>In top-k of %{customdata[0]} / " + str(n)
                       + " methods<br>Direction agreement: %{customdata[1]:.0%}<extra></extra>"),
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.INK_2))
    fig.update_xaxes(tickformat="+.0%", zeroline=True, zerolinecolor=ui.AXIS,
                     title=f"← pushes toward {neg}   ·   pushes toward {pos} →")
    fig.update_layout(height=max(300, 30 * len(feats) + 80), showlegend=False)
    return fig


def method_bars(values: pd.Series, title: str, fmt: str = ".3f", higher_better: bool = True,
                include_random: bool = True) -> go.Figure:
    s = values.copy()
    if not include_random:
        s = s.drop(index="random", errors="ignore")
    s = s.sort_values(ascending=not higher_better)
    fig = go.Figure(go.Bar(
        x=[ui.method_label(m) for m in s.index], y=s.values,
        marker=dict(color=[ui.method_color(m) for m in s.index], line=dict(width=0),
                    pattern=dict(shape=["/" if m == "random" else "" for m in s.index], fgcolor="white", size=6)),
        text=[format(v, fmt) for v in s.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.INK_2),
        hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ui.INK)), height=360, showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(showgrid=False, tickfont=dict(color=ui.INK_2))
    return fig


def deletion_lines(del_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m in del_df.columns:
        fig.add_trace(go.Scatter(
            x=del_df.index, y=del_df[m], mode="lines+markers", name=ui.method_label(m),
            line=dict(color=ui.method_color(m), width=2, dash="dot" if m == "random" else "solid"),
            marker=dict(size=7, line=dict(color="white", width=1.5)),
            hovertemplate=ui.method_label(m) + "<br>%{x} features removed → Δ %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(height=400, hovermode="x unified",
                      xaxis_title="Top features removed (in each method's order)",
                      yaxis_title="Change in predicted probability")
    fig.update_xaxes(dtick=1)
    return fig


def short(m: str) -> str:
    return METHOD_SHORT.get(m, m)
