"""Render the poster figures as PNGs from the project's own code and results.

    python3 scripts/make_poster_figures.py

Writes to docs/poster/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xaudit.audit import audit_dataset, audit_instance  # noqa: E402
from xaudit.benchmark import summarise_benchmark  # noqa: E402
from xaudit.config import METHOD_LABELS, METHODS, METHOD_COLORS, MODEL_LABELS  # noqa: E402
from xaudit.data import load_dataset  # noqa: E402
from xaudit.models import train_model  # noqa: E402

OUT = ROOT / "docs" / "poster" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLORS = {"logistic_regression": "#4a3aa7", "random_forest": "#008300",
                "gradient_boosting": "#e34948", "mlp": "#52514e"}
DIVERGING = LinearSegmentedColormap.from_list("xaudit", ["#1c5cab", "#6da7ec", "#f0efec", "#ee8a89", "#b8302f"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15,
    "text.color": INK2,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def fig_fingerprint(tm, index: int, k: int = 5) -> dict:
    """Features x methods heatmap with each method's rank in the cell."""
    a = audit_instance(tm, index, k=k)
    cons = a.consensus
    M, ranks = cons["normalised"], cons["ranks"]
    names = a.feature_names
    order = np.argsort(-np.abs(cons["mean"]))[:9]
    z = M[:, order].T
    lim = float(np.abs(z).max())

    fig, ax = plt.subplots(figsize=(11.5, 6.9), dpi=200)
    im = ax.imshow(z, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    short = {"shapley": "Shapley", "lime": "LIME", "integrated_gradients": "Integrated\nGradients",
             "smoothgrad_x_input": "SmoothGrad\n× Input", "occlusion": "Occlusion"}
    ax.set_xticklabels([short[m] for m in METHODS], fontsize=16, color=INK, linespacing=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([("* " if cons["contested"][j] else "   ") + names[j] for j in order],
                       fontsize=15, color=INK2)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(np.arange(-.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(METHODS)):
        for jj, j in enumerate(order):
            val = M[i, j]
            ax.text(i, jj, f"#{ranks[i, j] + 1}", ha="center", va="center", fontsize=15,
                    color="white" if abs(val) > 0.55 * lim else INK2,
                    fontweight="bold" if ranks[i, j] < k else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("share of explanation  (red → Default, blue → Repaid)", fontsize=13, color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fingerprint.png", bbox_inches="tight")
    plt.close(fig)
    return {"trust": a.trust_score, "level": a.trust_level, "prob": a.probability, "index": index}


def fig_trust_vs_truth(bench: pd.DataFrame, r: float, inst: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.1), dpi=200)
    ax.scatter(inst["trust"], inst["precision"], s=55, color="#2a78d6", alpha=0.45,
               edgecolor="white", linewidth=1.2, zorder=3)
    b1, b0 = np.polyfit(inst["trust"], inst["precision"], 1)
    xs = np.linspace(inst["trust"].min(), inst["trust"].max(), 20)
    ax.plot(xs, b0 + b1 * xs, color=INK, linewidth=3, zorder=4)
    ax.set_xlabel("Explanation Trust Score  (how much methods agree)", fontsize=16)
    ax.set_ylabel("True-driver precision  (how correct they are)", fontsize=16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=30, fontweight="bold",
            color=INK, va="top")
    ax.text(0.03, 0.86, f"{len(inst)} predictions · 5 methods each", transform=ax.transAxes,
            fontsize=14, color=MUTED, va="top")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "trust_vs_truth.png", bbox_inches="tight")
    plt.close(fig)


def fig_trust_vs_rho(trust: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=200)
    for mk in trust.columns:
        ax.plot(trust.index, trust[mk], marker="o", markersize=9, linewidth=3,
                color=MODEL_COLORS[mk], label=MODEL_LABELS[mk],
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(MODEL_LABELS[mk], (trust.index[-1], trust[mk].iloc[-1]), xytext=(10, 0),
                    textcoords="offset points", fontsize=14, color=MODEL_COLORS[mk],
                    va="center", fontweight="bold")
    ax.set_xlabel("Feature correlation  ρ", fontsize=16)
    ax.set_ylabel("Mean Explanation Trust Score", fontsize=16)
    ax.set_xlim(-0.04, 1.32)
    ax.set_xticks(list(trust.index))
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "trust_vs_rho.png", bbox_inches="tight")
    plt.close(fig)


def fig_method_precision(prec: pd.DataFrame) -> None:
    s = prec.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=200)
    bars = ax.bar([METHOD_LABELS[m].replace(" (sampling)", "").replace(" × Input", "×Input") for m in s.index],
                  s.values, color=[METHOD_COLORS[m] for m in s.index], width=0.62)
    for b, v in zip(bars, s.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=16,
                color=INK, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("True-driver precision", fontsize=16)
    ax.tick_params(axis="x", labelsize=14, colors=INK)
    ax.xaxis.grid(False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "method_precision.png", bbox_inches="tight")
    plt.close(fig)


def fig_model_trust(tm_dataset: str = "credit_risk", n: int = 30) -> dict:
    """Mean trust per model family on the credit-risk data, with accuracy underneath."""
    ds = load_dataset(tm_dataset)
    rows = {}
    for mk in MODEL_LABELS:
        tm = train_model(ds, mk)
        g = audit_dataset(tm, n=n, k=5)
        rows[mk] = {"trust": g.per_instance["trust_score"].mean(),
                    "sd": g.per_instance["trust_score"].std(),
                    "auc": tm.metrics["roc_auc"]}
    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=200)
    bars = ax.bar([MODEL_LABELS[m].replace("Neural Network (MLP)", "Neural Net") for m in df.index],
                  df["trust"], yerr=df["sd"], capsize=6, width=0.62,
                  color=[MODEL_COLORS[m] for m in df.index],
                  error_kw=dict(ecolor=AXIS, lw=2))
    for b, (_, r) in zip(bars, df.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, 4, f"{r['trust']:.0f}", ha="center", fontsize=20,
                color="white", fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, -9, f"AUC {r['auc']:.2f}", ha="center", fontsize=13,
                color=MUTED)
    ax.axhline(70, color="#0ca30c", linestyle=":", linewidth=2)
    ax.text(3.55, 71.5, "High trust", color="#0ca30c", fontsize=12, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean Explanation Trust Score", fontsize=16)
    ax.tick_params(axis="x", labelsize=14, colors=INK)
    ax.xaxis.grid(False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "model_trust.png", bbox_inches="tight")
    plt.close(fig)
    return df.to_dict(orient="index")


def main() -> None:
    print("1/5 training model and finding a contested prediction…")
    ds = load_dataset("credit_risk")
    tm = train_model(ds, "gradient_boosting")
    g = audit_dataset(tm, n=30, k=5)
    worst = int(g.per_instance.sort_values("trust_score").iloc[0]["test_index"])
    info = fig_fingerprint(tm, worst)
    print("   fingerprint:", info)

    print("2/5 benchmark summaries…")
    bench = pd.read_csv(ROOT / "results" / "benchmark.csv")
    s = summarise_benchmark(bench)
    print("3/5 trust vs truth scatter…")
    fig_trust_vs_truth(bench, s["trust_precision_corr"], s["instance_level"])
    print("4/5 trust vs rho + method precision…")
    fig_trust_vs_rho(s["trust"])
    fig_method_precision(s["precision"])
    print("5/5 model-family comparison…")
    print("   ", fig_model_trust())
    print("saved to", OUT)


if __name__ == "__main__":
    main()
