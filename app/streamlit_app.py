"""XAI Disagreement Auditor - Streamlit dashboard entry point.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
for p in (str(ROOT), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="XAI Disagreement Auditor",
    page_icon=":material/balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

import state  # noqa: E402
import ui  # noqa: E402
from views import faithfulness_view, global_audit, ground_truth, instance, methods, overview  # noqa: E402
from xaudit import __version__  # noqa: E402
from xaudit.config import DEFAULT_SEED, DEFAULT_TOP_K, MODEL_LABELS  # noqa: E402
from xaudit.data import BUILTIN_LABELS, DataValidationError, prepare_uploaded  # noqa: E402

ui.inject_css()

PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/space_dashboard:", url_path="overview",
                        default=True),
    "instance": st.Page(instance.render, title="Instance Audit", icon=":material/search:", url_path="instance"),
    "global": st.Page(global_audit.render, title="Global Audit", icon=":material/public:", url_path="global"),
    "faith": st.Page(faithfulness_view.render, title="Who's Right?", icon=":material/verified:",
                     url_path="faithfulness"),
    "gt": st.Page(ground_truth.render, title="Ground-Truth Lab", icon=":material/science:", url_path="lab"),
    "methods": st.Page(methods.render, title="Methods & Metrics", icon=":material/menu_book:", url_path="methods"),
}
st.session_state["pages"] = PAGES


def sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>XAI Disagreement Auditor</div>"
                    f"<div class='sidebar-sub'>AIPI 590 · Explainable AI · v{__version__}</div>",
                    unsafe_allow_html=True)
        st.write("")
        prev = state.get_config()

        options = list(BUILTIN_LABELS) + ["__upload__"]
        labels = {**BUILTIN_LABELS, "__upload__": "Upload your own CSV…"}
        current = "__upload__" if prev.dataset_key.startswith("uploaded:") else prev.dataset_key
        choice = st.selectbox("Dataset", options, index=options.index(current), format_func=labels.get)

        dataset_key = choice
        if choice == "__upload__":
            dataset_key = _upload_widget(prev)

        model_key = st.selectbox("Model to explain", list(MODEL_LABELS), index=list(MODEL_LABELS).index(prev.model_key),
                                 format_func=MODEL_LABELS.get)
        k = st.slider("Top-k features compared", 3, 10, prev.k,
                      help="Agreement metrics compare each method's k most important features.")
        with st.expander("Advanced"):
            n_global = st.slider("Predictions in global audit", 10, 150, prev.n_global, step=10)
            seed = st.number_input("Random seed", 0, 10_000, prev.seed, step=1)

        st.session_state["cfg"] = state.Config(dataset_key, model_key, int(k), int(seed), int(n_global))

        if not dataset_key.startswith("uploaded:") or "uploaded_ds" in st.session_state:
            cfg = st.session_state["cfg"]
            try:
                tm = state.get_model(cfg)
                m = tm.metrics
                st.markdown(
                    f"<div style='margin-top:10px;padding:12px 14px;border-radius:12px;background:white;"
                    f"border:1px solid {ui.GRID};font-size:.85rem;color:{ui.INK_2}'>"
                    f"<b style='color:{ui.INK}'>{tm.label}</b><br>"
                    f"Test accuracy <b>{m['accuracy']:.1%}</b> · AUC <b>{m['roc_auc']:.3f}</b><br>"
                    f"{tm.dataset.n_features} features · {m['n_test']} test rows</div>",
                    unsafe_allow_html=True)
            except Exception as e:  # pragma: no cover - surfaced to user
                st.error(f"Could not train model: {e}")


def _upload_widget(prev: state.Config) -> str:
    f = st.file_uploader("CSV file", type=["csv"], help="Binary or multi-class target; numeric and categorical "
                                                        "columns are handled automatically.")
    if f is None:
        return prev.dataset_key if prev.dataset_key.startswith("uploaded:") else "uploaded:none"
    raw = f.getvalue()
    try:
        df = pd.read_csv(f)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return "uploaded:none"
    cols = list(df.columns)
    guess = next((i for i, c in enumerate(cols) if str(c).lower() in {"target", "label", "y", "class", "outcome"}),
                 len(cols) - 1)
    target = st.selectbox("Target column", cols, index=guess)
    values = sorted(df[target].dropna().astype(str).unique().tolist())
    if len(values) > 50:
        st.error("Target has more than 50 distinct values - pick a categorical target.")
        return "uploaded:none"
    positive = st.selectbox("Positive class", values, index=len(values) - 1)
    key = state.upload_key(raw, target, positive)
    if st.session_state.get("uploaded_key") != key:
        try:
            ds = prepare_uploaded(df, target, positive, name=f"{f.name} → {target} = {positive}")
        except DataValidationError as e:
            st.error(str(e))
            return "uploaded:none"
        st.session_state["uploaded_ds"] = ds
        st.session_state["uploaded_key"] = key
        for n in ds.notes:
            st.info(n)
    return key


if "cfg" not in st.session_state:
    st.session_state["cfg"] = state.Config("credit_risk", "gradient_boosting", DEFAULT_TOP_K, DEFAULT_SEED, 40)

sidebar()
nav = st.navigation(list(PAGES.values()), position="sidebar")
nav.run()
st.markdown("<div class='footer'>XAI Disagreement Auditor · AIPI 590 Emerging Trends in Explainable AI · "
            "Duke University · Fall 2026</div>", unsafe_allow_html=True)
