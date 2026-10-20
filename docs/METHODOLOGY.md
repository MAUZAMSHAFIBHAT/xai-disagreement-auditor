# Methodology

## 1. Problem

Post-hoc feature attribution methods are widely used to explain black-box models. Different methods often give
**different explanations for the same prediction**: different top features, different rankings, and sometimes
opposite directions. Krishna et al. (2022) surveyed practitioners and found that most of them settle these
disagreements by gut feeling or by picking a favourite method. No principled per-prediction signal tells a user
whether an explanation can be trusted.

**Contribution of this project**

1. An **Explanation Trust Score** for every prediction, built from multiple agreement metrics, together with a
   list of the *contested features* driving any disagreement.
2. A **controlled ground-truth benchmark** that tests (a) how feature correlation and model family affect
   disagreement and (b) whether disagreement predicts that the explanations are *wrong*.
3. A **production-grade, reproducible tool**: dashboard, REST API, CLI and tests. It applies all of this to any
   tabular dataset, including user uploads.

## 2. Explanation methods

All methods explain `f(x) = P(y = 1 | x)` and return an attribution vector `φ ∈ ℝ^d` in probability units.
They share one background sample `B` (100 training rows), one reference point `μ` (training mean) and one
scale `σ` (training std). Because of this, disagreement comes from the methods, not from inconsistent setup.

| Method | Estimator |
|---|---|
| Shapley (sampling) | Štrumbelj & Kononenko permutation sampling with antithetic pairs; each permutation walks from a random background row to `x`. Unbiased for interventional Shapley values; `Σφ ≈ f(x) − E_B f`. |
| LIME | 1,500 samples `z = x + σ ⊙ ε`, `ε ~ N(0, I)`; kernel `exp(−‖ε‖² / (0.75√d)²)`; weighted ridge on `ε`. Attribution = `β_j (x_j − μ_j)/σ_j`, which puts slopes on the same scale as additive attributions. |
| Integrated Gradients | Straight path `μ → x` in 48 steps; each partial derivative is estimated with a forward difference whose step equals the path increment, so the method works for non-differentiable models. Exact on linear models; completeness is approximate otherwise. |
| SmoothGrad × Input | Central-difference gradient (step 0.1σ) averaged over 24 noisy copies (noise 0.15σ), times `(x − μ)`. |
| Occlusion | `φ_j = f(x) − E_{z∈B} f(x with x_j := z_j)` over 40 background rows. |

Unit tests verify that Shapley, IG, SmoothGrad × Input and Occlusion recover the exact contributions
`w_j (x_j − μ_j)` of a linear model. They also check that LIME recovers the linear coefficients and that our
Shapley estimates correlate > 0.99 with the reference `shap` library's LinearExplainer.

## 3. Agreement metrics (Krishna et al., 2022)

For methods `a, b` and top-k sets `T_a, T_b` (by |φ|):

- **Feature agreement** `|T_a ∩ T_b| / k`
- **Rank agreement**: share of positions `i ≤ k` where both methods put the same feature
- **Sign agreement** `|{j ∈ T_a ∩ T_b : sign φ_a,j = sign φ_b,j}| / k`
- **Signed rank agreement**: same feature, same position, same sign
- **Rank correlation**: Spearman ρ of |φ_a| and |φ_b| over all features
- **Pairwise rank agreement**: share of feature pairs `(i, j)` ordered the same way by both methods

An all-zero attribution vector (for example, a gradient on a flat region) is scored as zero agreement.

## 4. Explanation Trust Score

```
agree(a, b) = ( FeatureAgreement@k + SignAgreement@k + (1 + Spearman)/2 ) / 3
TrustScore  = 100 × mean over the 10 method pairs of agree(a, b)
```

The three terms cover *which* features matter, *which way* they push, and the *whole ranking*.

Bands: **High ≥ 70**, **Medium 50–70**, **Low < 50**. The bands were set so that linear models on the built-in
datasets, where the methods provably agree, land in High, and tree ensembles on highly correlated data land in
Medium or Low.

A feature is **contested** if it is in the top-k of some methods but not all, or if methods that rank it top-k
disagree on its sign. The **consensus explanation** is the mean of the L1-normalised attribution vectors.

## 5. Faithfulness

Following OpenXAI (Agarwal et al., 2022), with marginal ablation (replace features with values from 50
background rows and average):

- **PGI@k** = |f(x) − E f(x with top-k removed)|. Higher is better.
- **PGU@k** = |f(x) − E f(x with all but top-k removed)|. Lower is better.
- **Deletion curve**: PGI for k = 1…min(d, 10).

A random attribution vector is always included as a control.

*Known bias:* marginal ablation is the same operation Occlusion uses, so these tests favour Occlusion. We
report this in the dashboard and rely on ground truth for method ranking.

## 6. Ground-truth benchmark

- `X ~ N(0, Σ)` with `Σ_ii = 1`, `Σ_ij = ρ`, d = 8, n = 1,500, ρ ∈ {0, 0.3, 0.6, 0.8, 0.95}
- `y ~ Bernoulli(σ(2.0 x1 − 1.6 x2 + 1.2 x3 − 0.9 x4))`, so only 4 features matter
- Models: logistic regression, random forest, histogram gradient boosting, MLP (75/25 split)
- 20 random test points per (ρ, model), each explained by all 5 methods
- **True-driver precision** = |top-4(φ) ∩ {x1..x4}| / 4

The **Credit Risk** dataset uses the same idea with realistic feature names: 5 real drivers, 3 correlated proxies
(ρ = 0.55) and 2 pure-noise columns.

### Results (seed 0, 400 predictions, 2,000 explanations)

| | ρ = 0 | ρ = 0.95 |
|---|---|---|
| Mean trust: Logistic Regression | 93.5 | 95.8 |
| Mean trust: Random Forest | 77.1 | 67.3 |
| Mean trust: Gradient Boosting | 76.2 | 65.9 |
| Mean trust: MLP | 86.3 | 75.7 |
| Mean true-driver precision (all methods) | 0.86 | 0.68 |

Instance-level correlation between Trust Score and mean true-driver precision: **r = +0.61**.

Mean true-driver precision by method: LIME 0.84, Shapley 0.81, IG 0.79, Occlusion 0.76, SmoothGrad × Input 0.74.

## 7. Limitations and future work

- Agreement ≠ truth: all methods can agree on a proxy the model genuinely uses. The trust score is a *warning
  signal*, and the benchmark measures how reliable that signal is.
- Perturbation-based methods create off-manifold inputs under strong correlation. Conditional or
  on-manifold Shapley variants could be added.
- Supports tabular binary classification only. Future work: text and image models, and sampling variance as an
  additional "self-agreement" term (running each method twice with different seeds).
- The trust bands are heuristic. They could be calibrated against ground truth with isotonic regression on the
  benchmark data.

## 8. Stakeholders, misuse risks and deployment

Summarised here; the full treatment is in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

* **Who it is for.** Data scientists choosing an attribution method; model-risk and compliance teams deciding
  when a single explanation may be presented as the reason for a decision; auditors checking whether a stated
  explanation is method-dependent. It is not an end-user-facing system.
* **Explanation-washing** is the main misuse risk. A high Trust Score certifies that methods *concur*, not that
  the model is accurate, fair or lawful. Every surface that reports the score says so.
* **Score-shopping.** k, the background sample, the seed and the method set all move the number, so every run
  is seeded and the exported JSON records the full configuration. A score quoted without its configuration is
  unsupported.
* **Proxies raise the score.** If a model genuinely relies on a proxy, faithful methods agree on it and trust
  is *high*. Agreement is therefore not evidence of fairness; fairness must be tested separately.
* **Deployment.** Compute the score alongside the explanation, store both, and route low-trust predictions to
  human review. Everything runs locally; uploaded data is never written to disk and no network call is made at
  runtime.

## References

1. Krishna et al. (2022). The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective. arXiv:2202.01602.
2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
3. Štrumbelj & Kononenko (2014). Explaining prediction models and individual predictions with feature contributions. KAIS.
4. Ribeiro, Singh & Guestrin (2016). "Why Should I Trust You?" KDD.
5. Sundararajan, Taly & Yan (2017). Axiomatic Attribution for Deep Networks. ICML.
6. Smilkov et al. (2017). SmoothGrad: removing noise by adding noise. arXiv:1706.03825.
7. Shrikumar et al. (2016). Not Just a Black Box. arXiv:1605.01713.
8. Zeiler & Fergus (2014). Visualizing and Understanding Convolutional Networks. ECCV.
9. Agarwal et al. (2022). OpenXAI: Towards a Transparent Evaluation of Model Explanations. NeurIPS Datasets & Benchmarks.
