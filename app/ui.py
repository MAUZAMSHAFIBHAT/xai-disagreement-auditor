"""Shared look & feel: CSS, colour tokens, Plotly defaults and small HTML components."""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from xaudit.config import METHOD_COLORS, METHOD_LABELS

# ---- colour tokens (validated light palette) -------------------------------- #
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
BLUE = "#2a78d6"
RED = "#e34948"
NEUTRAL_MID = "#f0efec"
STATUS = {"High": "#0ca30c", "Medium": "#fab219", "Low": "#d03b3b"}
STATUS_ICON = {"High": "✔", "Medium": "▲", "Low": "✖"}
MODEL_COLORS = {
    "logistic_regression": "#4a3aa7",
    "random_forest": "#008300",
    "gradient_boosting": "#e34948",
    "mlp": "#52514e",
}
# sequential blue ramp (100 -> 700)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# diverging blue <-> gray <-> red
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, NEUTRAL_MID], [0.75, "#ee8a89"], [1.0, "#b8302f"]]


def method_color(m: str) -> str:
    return METHOD_COLORS.get(m, MUTED)


def method_label(m: str) -> str:
    return "Random (control)" if m == "random" else METHOD_LABELS.get(m, m)


# ---- Plotly template -------------------------------------------------------- #
_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[METHOD_COLORS[m] for m in METHOD_LABELS],
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=13)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)),
    margin=dict(l=10, r=10, t=40, b=10),
    bargap=0.25,
)
pio.templates["xaudit"] = _template
pio.templates.default = "plotly+xaudit"

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


def show(fig: go.Figure, height: int | None = None) -> None:
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ---- CSS -------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
[data-testid="stSidebar"] {{ background: #f3f2ee; border-right: 1px solid {GRID}; }}
[data-testid="stSidebar"] .sidebar-brand {{ font-weight: 800; font-size: 1.15rem; color: {INK}; letter-spacing: -0.02em; }}
[data-testid="stSidebar"] .sidebar-sub {{ color: {MUTED}; font-size: .8rem; margin-top: -2px; }}

.hero {{
  border-radius: 20px; padding: 34px 38px; margin-bottom: 22px; color: white;
  background: radial-gradient(1200px 400px at 0% 0%, #3987e5 0%, rgba(57,135,229,0) 60%),
              radial-gradient(900px 400px at 100% 100%, #4a3aa7 0%, rgba(74,58,167,0) 55%),
              linear-gradient(135deg, #0d366b 0%, #184f95 55%, #2a2466 100%);
  box-shadow: 0 10px 30px rgba(13,54,107,.18);
}}
.hero .eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; opacity: .85; font-weight: 600; }}
.hero h1 {{ color: white; font-size: 2.35rem; line-height: 1.15; margin: 8px 0 10px 0; font-weight: 800; }}
.hero p {{ font-size: 1.05rem; opacity: .92; max-width: 860px; margin: 0; }}
.hero .chips {{ margin-top: 16px; display:flex; gap:8px; flex-wrap:wrap; }}
.hero .chip {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); padding: 4px 12px;
  border-radius: 999px; font-size: .8rem; font-weight: 500; }}

.page-title {{ margin-bottom: 4px; }}
.page-title h2 {{ margin: 0; font-weight: 800; font-size: 1.9rem; }}
.page-title p {{ color: {INK_2}; margin: 4px 0 0 0; font-size: 1rem; }}

.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(11,11,11,.04); height: 100%; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1rem; color: {INK}; font-weight: 700; }}
.card p {{ margin: 0; color: {INK_2}; font-size: .92rem; line-height: 1.45; }}
.card .step {{ display:inline-flex; align-items:center; justify-content:center; width: 28px; height: 28px;
  border-radius: 8px; background: #e8f1fc; color: {BLUE}; font-weight: 800; margin-bottom: 10px; }}

.stat {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(11,11,11,.04); height: 100%; }}
.stat .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.stat .value {{ color: {INK}; font-size: 2rem; font-weight: 800; line-height: 1.15; margin-top: 4px; letter-spacing: -0.02em; }}
.stat .value.small {{ font-size: 1.25rem; padding-top: 8px; word-break: break-word; }}
.stat .sub {{ color: {INK_2}; font-size: .85rem; margin-top: 4px; }}

.badge {{ display:inline-flex; align-items:center; gap:6px; padding: 3px 10px; border-radius: 999px;
  font-size: .8rem; font-weight: 700; border: 1px solid; }}

.trust {{ border-radius: 18px; padding: 20px 22px; border: 1px solid rgba(11,11,11,.08); background: {SURFACE};
  box-shadow: 0 1px 2px rgba(11,11,11,.04); height: 100%; }}
.trust .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.trust .score {{ font-size: 3.2rem; font-weight: 800; color: {INK}; line-height: 1; margin: 8px 0 8px 0; letter-spacing: -0.03em; }}
.trust .score span {{ font-size: 1.1rem; color: {MUTED}; font-weight: 600; }}
.trust .bar {{ height: 8px; border-radius: 999px; background: #ecebe6; overflow: hidden; margin: 10px 0 6px 0; position: relative; }}
.trust .bar > div {{ height: 100%; border-radius: 999px; }}
.trust .ticks {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:.72rem; }}

.finding {{ border-left: 4px solid {BLUE}; background: #f2f7fd; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;
  color: {INK}; font-size: .95rem; }}
.finding.warn {{ border-left-color: #fab219; background: #fff8e6; }}
.finding.bad {{ border-left-color: #d03b3b; background: #fdf0f0; }}
.finding.good {{ border-left-color: #0ca30c; background: #eef8ee; }}

.legend-chip {{ display:inline-flex; align-items:center; gap:6px; margin-right: 14px; font-size: .85rem; color: {INK_2}; }}
.legend-chip i {{ width: 12px; height: 12px; border-radius: 3px; display:inline-block; }}

.section {{ margin-top: 18px; margin-bottom: 4px; }}
.section h3 {{ font-size: 1.2rem; font-weight: 700; margin: 0; }}
.section p {{ color: {INK_2}; margin: 2px 0 0 0; font-size: .92rem; }}

div[data-testid="stTabs"] button p {{ font-weight: 600; }}
div[data-testid="stExpander"] {{ border-radius: 12px; }}
.footer {{ color: {MUTED}; font-size: .8rem; margin-top: 30px; text-align: center; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---- components ------------------------------------------------------------- #
def page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-title'><h2>{html.escape(title)}</h2>"
                f"<p>{html.escape(subtitle)}</p></div>", unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"<div class='section'><h3>{html.escape(title)}</h3>{sub}</div>", unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "", small: bool = False) -> str:
    cls = "value small" if small else "value"
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    return (f"<div class='stat'><div class='label'>{html.escape(label)}</div>"
            f"<div class='{cls}'>{html.escape(value)}</div>{sub_html}</div>")


def stat_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        c.markdown(stat(*item), unsafe_allow_html=True)


def badge(level: str) -> str:
    col = STATUS[level]
    return (f"<span class='badge' style='color:{col};border-color:{col};background:{col}14'>"
            f"{STATUS_ICON[level]} {level} trust</span>")


def trust_card(score: float, level: str, caption: str = "") -> str:
    col = STATUS[level]
    return f"""
<div class='trust'>
  <div class='label'>Explanation Trust Score</div>
  <div class='score'>{score:.0f}<span> / 100</span></div>
  {badge(level)}
  <div class='bar'><div style='width:{max(2, min(100, score)):.0f}%;background:{col}'></div></div>
  <div class='ticks'><span>0</span><span>Low &lt; 50</span><span>High ≥ 70</span><span>100</span></div>
  <div style='color:{INK_2};font-size:.85rem;margin-top:10px'>{caption}</div>
</div>"""


def finding(text_md: str, kind: str = "") -> None:
    import re
    t = html.escape(text_md)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    st.markdown(f"<div class='finding {kind}'>{t}</div>", unsafe_allow_html=True)


def method_legend(methods: list[str]) -> None:
    chips = "".join(f"<span class='legend-chip'><i style='background:{method_color(m)}'></i>{method_label(m)}</span>"
                    for m in methods)
    st.markdown(f"<div style='margin:4px 0 8px 0'>{chips}</div>", unsafe_allow_html=True)


def card(title: str, body: str, step: str | None = None) -> str:
    s = f"<div class='step'>{step}</div>" if step else ""
    return f"<div class='card'>{s}<h4>{html.escape(title)}</h4><p>{body}</p></div>"
