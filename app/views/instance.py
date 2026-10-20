"""Instance Audit: five explanations of one prediction, and how much they agree."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import charts
import state
import ui
from xaudit.agreement import METRIC_HELP, METRIC_LABELS, METRICS, pair_score


def _pick_instance(cfg, tm) -> int:
    n = len(tm.X_test)
    if "focus_index" not in st.session_state or st.session_state["focus_index"] >= n:
        st.session_state["focus_index"] = 0
    g = state.get_global_audit(cfg, compute=False)

    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    with c1:
        idx = st.number_input(f"Test-set row (0 – {n - 1})", 0, n - 1, int(st.session_state["focus_index"]), step=1)
        st.session_state["focus_index"] = int(idx)
    with c2:
        st.write("")
        if st.button("Random row", icon=":material/casino:", use_container_width=True):
            st.session_state["focus_index"] = int(np.random.default_rng().integers(n))
            st.rerun()
    with c3:
        st.write("")
        if st.button("Most contested", icon=":material/warning:", use_container_width=True,
                     disabled=g is None, help="Run the Global Audit first to unlock." if g is None else None):
            st.session_state["focus_index"] = int(g.per_instance.sort_values("trust_score").iloc[0]["test_index"])
            st.rerun()
    with c4:
        st.write("")
        if st.button("Most agreed", icon=":material/handshake:", use_container_width=True,
                     disabled=g is None, help="Run the Global Audit first to unlock." if g is None else None):
            st.session_state["focus_index"] = int(g.per_instance.sort_values("trust_score").iloc[-1]["test_index"])
            st.rerun()
    return int(st.session_state["focus_index"])


def _what_if(tm, idx: int) -> np.ndarray | None:
    with st.expander("What-if mode: change feature values and watch the explanations react", icon=":material/tune:"):
        on = st.toggle("Use edited input instead of the original row", key="whatif_on")
        x = tm.X_test[idx].copy()
        names = tm.feature_names
        base = state.get_instance_audit(state.get_config(), idx)
        top = list(np.argsort(-np.abs(base.consensus["mean"]))[:6])
        cols = st.columns(3)
        lo = np.percentile(tm.X_train, 1, axis=0)
        hi = np.percentile(tm.X_train, 99, axis=0)
        for n_, j in enumerate(top):
            a, b = float(min(lo[j], x[j])), float(max(hi[j], x[j]))
            if a == b:
                continue
            step = float((b - a) / 100)
            x[j] = cols[n_ % 3].slider(names[j], a, b, float(x[j]), step=step, key=f"wi_{idx}_{j}",
                                       format="%.3f" if b - a < 10 else "%.0f")
        return x if on else None


def render() -> None:
    cfg = state.get_config()
    tm = state.get_model(cfg)
    ds = tm.dataset
    ui.page_title("Instance Audit", f"Five explanations of one {tm.label} prediction on {state.dataset_label(cfg)}, "
                                    f"and how far you can trust them.")

    idx = _pick_instance(cfg, tm)
    x_custom = _what_if(tm, idx)
    with st.spinner("Running 5 explanation methods…"):
        audit = state.get_instance_audit(cfg, idx, x_custom)

    # ---- headline row --------------------------------------------------------
    left, mid, right = st.columns([1, 1.05, 1.9])
    p = audit.probability
    pred_col = ui.RED if p >= 0.5 else ui.BLUE
    correct = "" if x_custom is not None else (
        "✔ correct" if audit.predicted_label == audit.true_label else "✖ misclassified")
    left.markdown(f"""
<div class='trust'>
  <div class='label'>Model prediction</div>
  <div class='score' style='font-size:2.4rem;margin-top:12px'>{p:.0%}<span>&nbsp;P({ds.positive_label})</span></div>
  <div style='font-weight:700;color:{pred_col};font-size:1.05rem'>→ {audit.predicted_label}</div>
  <div class='bar'><div style='width:{max(2, p * 100):.0f}%;background:{pred_col}'></div></div>
  <div style='color:{ui.INK_2};font-size:.85rem;margin-top:8px'>True label: <b>{audit.true_label}</b>
  &nbsp;{correct}</div>
</div>""", unsafe_allow_html=True)
    caption = {"High": "Methods broadly agree. The explanation is safe to communicate.",
               "Medium": "Partial agreement. Report only the features all methods share.",
               "Low": "Methods tell different stories. Do not present a single explanation as fact."}
    mid.markdown(ui.trust_card(audit.trust_score, audit.trust_level, caption[audit.trust_level]),
                 unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='label' style='color:{ui.MUTED};font-size:.78rem;text-transform:uppercase;"
                    f"letter-spacing:.08em;font-weight:600;margin-bottom:8px'>What the auditor found</div>",
                    unsafe_allow_html=True)
        for i, line in enumerate(audit.summary[:4]):
            kind = "warn" if "conflict" in line.lower() else ("good" if line.startswith("All") or
                                                                line.startswith("Safe") else "")
            ui.finding(line, kind)

    # ---- fingerprint ---------------------------------------------------------
    ui.section("Explanation fingerprint",
               "Each cell is a feature's share of that method's explanation (red pushes toward "
               f"<b>{ds.positive_label}</b>, blue toward <b>{ds.negative_label}</b>). The number is the method's "
               "rank for that feature. ⚠ marks contested features.")
    ui.show(charts.fingerprint(audit, pos=ds.positive_label, neg=ds.negative_label))

    tabs = st.tabs(["Side by side", "Agreement matrix", "Pairwise metrics", "Consensus explanation",
                    "Faithfulness", "Ground truth" if ds.true_weights else "Raw values"])
    with tabs[0]:
        st.caption("The same features in the same order, as each method sees them. Bars are shares of each "
                   "method's total attribution, so methods are comparable.")
        ui.show(charts.side_by_side(audit))
    with tabs[1]:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            ui.show(charts.agreement_matrix(audit.report["matrix"], audit.report["methods"]))
        with c2:
            st.markdown("**How the pair agreement is computed**")
            st.caption("Average of three things: do they pick the same top-k features, do those features push "
                       "the same way, and how well do their full rankings correlate. The Trust Score is the "
                       "mean over all 10 method pairs.")
            mm = audit.report["mean_metrics"]
            st.dataframe(pd.DataFrame({"Metric": [METRIC_LABELS[m] for m in METRICS],
                                       "Mean over pairs": [mm[m] for m in METRICS],
                                       "Meaning": [METRIC_HELP[m] for m in METRICS]}),
                         hide_index=True, use_container_width=True,
                         column_config={"Mean over pairs": st.column_config.ProgressColumn(
                             min_value=0.0, max_value=1.0, format="%.2f")})
    with tabs[2]:
        rows = []
        for (a, b), m in audit.report["pairs"].items():
            rows.append({"Method A": ui.method_label(a), "Method B": ui.method_label(b),
                         "Agreement": pair_score(m), **{METRIC_LABELS[k]: v for k, v in m.items()}})
        df = pd.DataFrame(rows).sort_values("Agreement")
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"Agreement": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                 format="%.2f")})
    with tabs[3]:
        st.caption("Mean of the five normalised explanations. Hatched bars are contested features; hover to see "
                   "how many methods have the feature in their top-k and how much they agree on its direction.")
        ui.show(charts.consensus_bars(audit, pos=ds.positive_label, neg=ds.negative_label))
    with tabs[4]:
        fa = audit.faithfulness.set_index("method")
        c1, c2 = st.columns(2)
        with c1:
            ui.show(charts.method_bars(fa["pgi"], "PGI: prediction change when removing top-k (higher = better)"))
        with c2:
            ui.show(charts.method_bars(fa["pgu"], "PGU: change when keeping only top-k (lower = better)",
                                       higher_better=False))
        st.caption("Hatched grey bar = random attributions, the floor every real method should beat.")
    with tabs[5]:
        if ds.true_weights and audit.ground_truth is not None:
            gt = audit.ground_truth.copy()
            gt["method"] = gt["method"].map(ui.method_label)
            gt["noise_features_in_top"] = gt["noise_features_in_top"].map(lambda v: ", ".join(v) or "none")
            st.caption("This dataset is synthetic, so the real drivers are known. True-driver precision is the share "
                       "of a method's top features that genuinely drive the label.")
            st.dataframe(gt.rename(columns={
                "method": "Method", "true_driver_precision": "True-driver precision",
                "truth_rank_correlation": "Rank corr. with truth", "direction_accuracy": "Direction accuracy",
                "noise_features_in_top": "Noise features it ranked highly"}),
                hide_index=True, use_container_width=True,
                column_config={"True-driver precision": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                        format="%.2f")})
        else:
            st.dataframe(audit.attribution_frame().round(4), use_container_width=True)

    # ---- export --------------------------------------------------------------
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.download_button("JSON report", json.dumps(audit.to_dict(), indent=2, default=float),
                       file_name=f"xai_audit_row{idx}.json", mime="application/json", icon=":material/download:",
                       use_container_width=True)
    c2.download_button("CSV attributions", audit.attribution_frame().to_csv().encode(),
                       file_name=f"attributions_row{idx}.csv", mime="text/csv", icon=":material/table:",
                       use_container_width=True)
    t = sum(audit.timing.values())
    c3.caption(f"All five methods ran in {t:.2f}s · " + " · ".join(
        f"{charts.short(m)} {s * 1000:.0f} ms" for m, s in audit.timing.items()))
