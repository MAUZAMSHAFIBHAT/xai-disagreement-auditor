"""Methods & Metrics reference page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from xaudit.agreement import METRIC_HELP, METRIC_LABELS, METRICS
from xaudit.config import METHOD_FAMILY, METHODS, TRUST_HIGH, TRUST_LOW


def render() -> None:
    ui.page_title("Methods & Metrics", "Everything the auditor computes, and the papers it builds on.")

    ui.section("Explanation methods")
    detail = {
        "shapley": ("Permutation-sampling Shapley values (256 antithetic permutations) against 100 background rows.",
                    "Satisfies efficiency: attributions sum to f(x) − E[f].", "Slow for many features; "
                                                                               "breaks feature correlations."),
        "lime": ("1,500 Gaussian perturbations around x, exponential kernel, weighted ridge. Coefficients × "
                 "standardised deviation from the mean.", "Intuitive local linear story.",
                 "Sensitive to kernel width and sampling; unstable."),
        "integrated_gradients": ("48-step Riemann path from the training mean to x with finite-difference partials.",
                                 "Completeness (approximately) and implementation invariance.",
                                 "Needs a baseline; gradients vanish on tree models."),
        "smoothgrad_x_input": ("Central-difference gradient averaged over 24 noisy copies (σ = 0.15 std), × (x − mean).",
                               "Cheap and purely local.", "Only sees tiny neighbourhoods; blind to thresholds."),
        "occlusion": ("Replace one feature with 40 background values; average drop in prediction.",
                      "Simple, model-agnostic, easy to explain.", "Ignores interactions between features."),
    }
    st.dataframe(pd.DataFrame([{"Method": ui.method_label(m), "Family": METHOD_FAMILY[m],
                                "Implementation": detail[m][0], "Strength": detail[m][1],
                                "Weakness": detail[m][2]} for m in METHODS]),
                 hide_index=True, use_container_width=True)
    st.caption("All five are implemented from scratch in NumPy (see xaudit/explainers.py) so they share exactly the "
               "same model, background data and baseline. Unit tests check Shapley efficiency, IG completeness and "
               "agreement with the reference `shap` library on linear models.")

    ui.section("Agreement metrics", "From Krishna et al. (2022). Each compares two methods' explanations.")
    st.dataframe(pd.DataFrame({"Metric": [METRIC_LABELS[m] for m in METRICS],
                               "Definition": [METRIC_HELP[m] for m in METRICS]}),
                 hide_index=True, use_container_width=True)

    ui.section("Explanation Trust Score")
    st.markdown(
        f"""
For each of the 10 method pairs we compute a **pair agreement** in [0, 1]:

$$\\text{{agree}}(a,b) = \\tfrac13\\Big(\\text{{FeatureAgreement}}@k + \\text{{SignAgreement}}@k + \\tfrac{{1+\\rho_{{\\text{{Spearman}}}}}}{{2}}\\Big)$$

The **Trust Score** is 100 × the mean over all pairs. It captures *which* features matter, *which way* they push,
and the *full ranking*. Bands: **High ≥ {TRUST_HIGH:.0f}**, **Medium**, **Low < {TRUST_LOW:.0f}**.

A feature is **contested** when some methods rank it in their top-k and others don't, or when methods that rank
it highly disagree on its direction.
""")

    ui.section("Faithfulness tests")
    st.markdown("""
* **PGI (Prediction Gap on Important features):** remove the top-k features and measure |Δ prediction|. Higher is better.
* **PGU (Prediction Gap on Unimportant features):** keep *only* the top-k. Lower is better.
* **Deletion curve:** remove features one by one in each method's order.
* "Remove" means replacing the feature with values drawn from the training data and averaging, not a single constant.
""")

    ui.section("Limitations")
    st.markdown("""
* The trust score measures **agreement, not truth**. All methods can agree and still be wrong, for example when the
  model relies on a proxy. The Ground-Truth Lab measures how often agreement *does* track truth.
* Removal-based faithfulness tests share Occlusion's removal operation, which favours Occlusion.
* Perturbation methods create off-manifold inputs when features are strongly correlated.
* Only tabular binary classification is supported. Multi-class uploads are converted to one-vs-rest.
""")

    ui.section("Stakeholders, misuse risks & deployment")
    st.markdown("""
**Who this is for.** Data scientists choosing an attribution method rather than defaulting to a habit;
model-risk and compliance teams deciding when a single explanation may be presented as *the* reason for a
decision; auditors checking whether a stated explanation is method-dependent. It is not an end-user-facing
system, and nothing here should be shown to an affected person as the reason for a decision.

**Explanation-washing — the main misuse risk.** A Trust Score of 90 certifies that the five methods *concur*.
It says nothing about whether the model is accurate, fair, lawful or appropriate. Quoting it as evidence of
"explainability" in a model card or a regulatory response is the abuse this project most plausibly enables.
It belongs *alongside* model validation, never instead of it.

**Score-shopping.** Top-k, the background sample, the seed and the method set all move the number. Mitigations:
fixed defaults, seeded and reproducible runs, and an exported JSON report that records the dataset, model, k,
seed and per-method timings. A score quoted without its configuration is unsupported.

**Proxies raise the score.** If a model genuinely relies on a proxy for a protected attribute, faithful methods
agree on that proxy and trust is *high*. Agreement is therefore not evidence of fairness — but it does make the
proxy easier to see, which is the constructive use.

**Deployment.** Compute the score alongside the explanation, store both, and route low-trust predictions to
human review rather than to an automated gate — the bands (70 / 50) are reasoned cut-offs, not calibrated
probabilities. Everything runs locally: uploaded data stays in memory, is never written to disk, and no network
call is made at runtime.

See `docs/RESPONSIBLE_USE.md` for the full treatment.
""")

    ui.section("References")
    st.markdown("""
1. Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S., & Lakkaraju, H. (2022). *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.* arXiv:2202.01602.
2. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
3. Štrumbelj, E., & Kononenko, I. (2014). *Explaining prediction models and individual predictions with feature contributions.* KAIS.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* KDD.
5. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
6. Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). *SmoothGrad: removing noise by adding noise.* arXiv:1706.03825.
7. Shrikumar, A., Greenside, P., Shcherbina, A., & Kundaje, A. (2016). *Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.* arXiv:1605.01713.
8. Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.* ECCV.
9. Agarwal, C., et al. (2022). *OpenXAI: Towards a Transparent Evaluation of Model Explanations.* NeurIPS Datasets & Benchmarks.
""")
