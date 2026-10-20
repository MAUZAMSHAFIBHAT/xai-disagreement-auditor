# Responsible use: stakeholders, misuse risks and deployment

The auditor exists to make explanations *more* trustworthy, which means it can also be used to make
untrustworthy work *look* trustworthy. This document states who the tool is for, how it can be abused, and
what has to be true before anyone relies on it in production.

## 1. Stakeholders

| Who | What they get | What they must not conclude |
|---|---|---|
| **Data scientists / ML engineers** | A reason to choose an attribution method for a given model and dataset, instead of defaulting to a habit; a per-prediction flag for "this explanation is unstable". | That a high score validates the model, or that the highest-scoring method is "correct". |
| **Model risk, audit and compliance teams** | Evidence about *when* a single explanation should not be presented as the reason for a decision; an exportable, configuration-stamped JSON report per decision. | That the score is a compliance artefact or a substitute for fairness testing, documentation or human review. |
| **Regulators and external auditors** | A reproducible, open procedure for asking whether a firm's stated explanation is method-dependent. | That agreement implies the model is lawful, fair or accurate. |
| **People affected by a decision** (loan applicant, patient) | Indirect benefit: a reason for a decision that does not depend on which library the analyst imported. | — |
| **The organisation deploying the model** | A pre-deployment signal that a model family will be hard to explain, while a choice between families is still open. | That the auditor reduces its obligation to justify individual decisions. |

The tool is aimed at the people producing and checking explanations. It is **not** an end-user-facing system,
and nothing in it should be shown to an affected person as "the reason" for a decision.

## 2. Misuse risks and mitigations

**Explanation-washing (most likely misuse).** A Trust Score of 90 says the five methods concur. It says nothing
about whether the model is accurate, fair, legal, or appropriate for the decision. The realistic abuse is
pasting a high score into a model card or a regulatory response as evidence of "explainability". Mitigations
in the tool: every page that reports the score also states that agreement is not truth; the Methods page says
so explicitly; the Ground-Truth Lab quantifies how often agreement does and does not track correctness
(r ≈ 0.61 — informative, far from certain).

**Score-shopping.** Top-k, the background sample, the random seed and the set of methods all move the number. A
team wanting a high score could tune them. Mitigations: fixed defaults in `xaudit/config.py`; every run is
seeded and reproducible; the exported JSON report records the dataset, model, k, seed and per-method timings,
so a reader can see the configuration that produced a number. A score quoted without its configuration should
be treated as unsupported.

**False reassurance on a model that uses a proxy.** When a model genuinely relies on a proxy for a protected
attribute, faithful explanations will agree on the proxy and the Trust Score will be *high*. High agreement
therefore cannot be read as evidence of fairness; if anything, it makes a proxy easier to see — which is the
constructive use. Fairness must be tested separately.

**Over-trusting the bands.** High / Medium / Low (70 / 50) are reasoned cut-offs, not calibrated probabilities.
Until they are calibrated (see future work), they should be used for triage, not as thresholds in an automated
control.

**Synthetic data mistaken for real evidence.** The Credit Risk dataset is generated, and is labelled as such
everywhere it appears. It exists so that ground truth is known. No claim about real lending follows from it.

## 3. Deployment considerations

* **Where it fits.** As a check in model validation and in explanation-generating pipelines: compute the Trust
  Score alongside the explanation, store both, and suppress or qualify single-method explanations for
  low-trust predictions.
* **Cost.** Five methods on one prediction take well under a second on a laptop for tabular models of this
  size, but cost grows with the number of features and the cost of a model call. Shapley sampling dominates.
  Budgets are tunable via `XAUDIT_*` environment variables.
* **Reproducibility.** Scores are seeded. The same prediction, model, seed and configuration give the same
  score. Change any of them and the score moves; report them together.
* **Data handling.** Everything runs locally; uploaded CSVs stay in the session's memory and are never written
  to disk or sent anywhere. There is no telemetry and no network call at runtime. A deployment handling
  personal data should still apply its own access controls — explanations can leak information about training
  data and about individuals.
* **Human in the loop.** The intended operating mode is: low trust → a human reviews before any explanation is
  communicated. The tool is a triage signal, not an automated gate.

## 4. What would make this safe to rely on

1. Calibrated bands (Trust Score → probability that the top-k features are genuinely the drivers).
2. Validation beyond tabular binary classification.
3. A user study testing whether practitioners actually act differently when shown a trust score — the
   disagreement problem was identified through practitioner interviews, and the remedy deserves the same
   scrutiny.

Until those exist, the honest claim is the one the poster makes: **disagreement is a useful warning signal,
and agreement is not proof.**
