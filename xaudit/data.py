"""Dataset loading.

Three built-in datasets ship with the project, so it runs fully offline:

* ``breast_cancer`` - Wisconsin Diagnostic Breast Cancer (scikit-learn), 30 features.
* ``diabetes`` - scikit-learn diabetes data, binarised at the median progression score.
* ``credit_risk`` - a synthetic loan-default dataset with a *known* data-generating
  process, so explanations can be checked against ground truth.

Users can also upload their own CSV (see :func:`prepare_uploaded`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from xaudit.config import MAX_FEATURES, MAX_UPLOAD_ROWS


@dataclass
class Dataset:
    """A tabular binary-classification dataset ready for modelling."""

    key: str
    name: str
    X: pd.DataFrame
    y: np.ndarray
    positive_label: str
    negative_label: str
    description: str
    # For synthetic data: true per-feature weights in the logit (0 = irrelevant).
    true_weights: dict[str, float] | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


# --------------------------------------------------------------------------- #
# Built-in datasets
# --------------------------------------------------------------------------- #

def _breast_cancer() -> Dataset:
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    # sklearn codes malignant=0, benign=1. Flip so the positive class is "malignant".
    y = (raw.target.values == 0).astype(int)
    return Dataset(
        key="breast_cancer",
        name="Breast Cancer Diagnosis",
        X=X,
        y=y,
        positive_label="Malignant",
        negative_label="Benign",
        description=(
            "569 tumour biopsies described by 30 cell-nucleus measurements. "
            "Many features are strongly correlated (radius, perimeter, area), "
            "which is exactly where explanation methods tend to disagree."
        ),
        source="Wolberg et al., UCI ML Repository (via scikit-learn)",
    )


def _diabetes() -> Dataset:
    raw = load_diabetes(as_frame=True, scaled=False)
    X = raw.data.copy()
    X = X.rename(columns={"s1": "total_cholesterol", "s2": "ldl", "s3": "hdl",
                          "s4": "chol_hdl_ratio", "s5": "log_triglycerides", "s6": "blood_sugar"})
    y = (raw.target.values > np.median(raw.target.values)).astype(int)
    return Dataset(
        key="diabetes",
        name="Diabetes Progression",
        X=X,
        y=y,
        positive_label="High progression",
        negative_label="Low progression",
        description=(
            "442 patients with 10 baseline measurements. The target is whether "
            "disease progression one year later is above the median."
        ),
        source="Efron et al. 2004 (via scikit-learn)",
    )


CREDIT_WEIGHTS: dict[str, float] = {
    # informative features (standardised-scale weights in the logit)
    "debt_to_income": 1.6,
    "num_late_payments": 1.3,
    "annual_income": -1.1,
    "credit_history_years": -0.8,
    "loan_amount": 0.6,
    # correlated-but-irrelevant and pure-noise features
    "monthly_spending": 0.0,
    "num_credit_cards": 0.0,
    "age": 0.0,
    "years_at_address": 0.0,
    "customer_id_hash": 0.0,
}


def make_credit_risk(n: int = 2500, seed: int = 7, rho: float = 0.55) -> Dataset:
    """Synthetic loan-default data with a known generating process.

    Irrelevant features are deliberately correlated with relevant ones
    (e.g. ``monthly_spending`` tracks ``annual_income``; ``age`` tracks
    ``credit_history_years``) so the model *can* lean on proxies. The true
    logit only uses the five features with non-zero weight.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 10))
    Z = pd.DataFrame(z, columns=list(CREDIT_WEIGHTS))
    s = np.sqrt(1 - rho**2)
    # correlated proxies
    Z["monthly_spending"] = rho * Z["annual_income"] + s * Z["monthly_spending"]
    Z["age"] = rho * Z["credit_history_years"] + s * Z["age"]
    Z["num_credit_cards"] = rho * Z["num_late_payments"] + s * Z["num_credit_cards"]
    Z["loan_amount"] = 0.4 * Z["annual_income"] + np.sqrt(1 - 0.16) * Z["loan_amount"]

    logit = sum(w * Z[c] for c, w in CREDIT_WEIGHTS.items()) - 0.4
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < np.asarray(p)).astype(int)

    # Put features on human-readable scales (monotone affine maps keep the truth intact).
    X = pd.DataFrame({
        "debt_to_income": np.clip(0.32 + 0.11 * Z["debt_to_income"], 0.01, None).round(3),
        "num_late_payments": np.clip(np.round(2 + 1.5 * Z["num_late_payments"]), 0, None),
        "annual_income": np.round(62000 + 18000 * Z["annual_income"], -2).clip(12000, None),
        "credit_history_years": np.clip(9 + 5 * Z["credit_history_years"], 0, None).round(1),
        "loan_amount": np.round(18000 + 7000 * Z["loan_amount"], -2).clip(1000, None),
        "monthly_spending": np.round(3100 + 900 * Z["monthly_spending"], -1).clip(300, None),
        "num_credit_cards": np.clip(np.round(3 + 1.4 * Z["num_credit_cards"]), 0, None),
        "age": np.clip(38 + 11 * Z["age"], 18, 90).round(0),
        "years_at_address": np.clip(6 + 4 * Z["years_at_address"], 0, None).round(1),
        "customer_id_hash": (50 + 29 * Z["customer_id_hash"]).round(2),
    })
    return Dataset(
        key="credit_risk",
        name="Credit Risk (synthetic, ground truth known)",
        X=X,
        y=y,
        positive_label="Default",
        negative_label="Repaid",
        description=(
            "2,500 synthetic loan applicants. Only 5 of the 10 features actually "
            "drive default risk; the rest are noise or correlated proxies. "
            "Because the truth is known, we can check which explanation method is right."
        ),
        true_weights=dict(CREDIT_WEIGHTS),
        source="Generated by xaudit.data.make_credit_risk (seeded, reproducible)",
    )


def make_ground_truth_benchmark(n: int, d: int, rho: float, n_informative: int, seed: int) -> Dataset:
    """Equicorrelated Gaussian features; only the first ``n_informative`` matter."""
    rng = np.random.default_rng(seed)
    cov = np.full((d, d), rho) + np.eye(d) * (1 - rho)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    base = np.array([2.0, -1.6, 1.2, -0.9, 0.7, -0.5, 0.4, -0.3])
    w = np.zeros(d)
    w[:n_informative] = np.resize(base, n_informative)
    logit = X @ w
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    names = [f"x{i + 1}{'*' if i < n_informative else ''}" for i in range(d)]
    return Dataset(
        key=f"gt_rho{rho:.2f}",
        name=f"Ground-truth benchmark (rho={rho:.2f})",
        X=pd.DataFrame(X, columns=names),
        y=y,
        positive_label="1",
        negative_label="0",
        description="Equicorrelated Gaussian benchmark; starred features are informative.",
        true_weights=dict(zip(names, w.tolist())),
        source="xaudit.data.make_ground_truth_benchmark",
    )


BUILTIN_DATASETS = {
    "credit_risk": make_credit_risk,
    "breast_cancer": _breast_cancer,
    "diabetes": _diabetes,
}

BUILTIN_LABELS = {
    "credit_risk": "Credit Risk (synthetic, ground truth known)",
    "breast_cancer": "Breast Cancer Diagnosis",
    "diabetes": "Diabetes Progression",
}


def load_dataset(key: str) -> Dataset:
    if key not in BUILTIN_DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Options: {sorted(BUILTIN_DATASETS)}")
    return BUILTIN_DATASETS[key]()


# --------------------------------------------------------------------------- #
# User uploads
# --------------------------------------------------------------------------- #

class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used."""


def prepare_uploaded(df: pd.DataFrame, target: str, positive_value: str | None = None,
                     name: str = "Uploaded dataset") -> Dataset:
    """Turn an arbitrary CSV into a numeric binary-classification dataset.

    * Rows with a missing target are dropped.
    * Numeric features: missing values filled with the median.
    * Categorical features with <= 15 levels: one-hot encoded; others dropped.
    * Multi-class targets: one-vs-rest on ``positive_value``.
    """
    if target not in df.columns:
        raise DataValidationError(f"Target column '{target}' not found.")
    df = df.dropna(subset=[target]).copy()
    if len(df) > MAX_UPLOAD_ROWS:
        df = df.sample(MAX_UPLOAD_ROWS, random_state=0)
    if len(df) < 50:
        raise DataValidationError("Need at least 50 rows with a target value.")

    y_raw = df.pop(target).astype(str)
    classes = sorted(y_raw.unique())
    if len(classes) < 2:
        raise DataValidationError("Target column has only one class.")
    if positive_value is None:
        positive_value = y_raw.value_counts().index[-1]  # minority class
    positive_value = str(positive_value)
    if positive_value not in classes:
        raise DataValidationError(f"'{positive_value}' is not a value of '{target}'.")
    y = (y_raw.values == positive_value).astype(int)
    neg_label = "Other" if len(classes) > 2 else next(c for c in classes if c != positive_value)

    notes: list[str] = []
    num = df.select_dtypes(include=[np.number])
    cat = df.drop(columns=num.columns)
    num = num.loc[:, num.nunique() > 1]
    num = num.fillna(num.median())
    keep_cat = [c for c in cat.columns if 1 < cat[c].nunique() <= 15]
    dropped = sorted(set(cat.columns) - set(keep_cat))
    if dropped:
        notes.append(f"Dropped high-cardinality or constant text columns: {', '.join(dropped[:8])}")
    parts = [num]
    if keep_cat:
        dummies = pd.get_dummies(cat[keep_cat].astype(str), prefix=keep_cat, drop_first=True, dtype=float)
        parts.append(dummies)
    X = pd.concat(parts, axis=1)
    X.columns = [str(c) for c in X.columns]
    if X.shape[1] == 0:
        raise DataValidationError("No usable feature columns after cleaning.")
    if X.shape[1] > MAX_FEATURES:
        # keep the most variable (standardised by mean) columns
        var = X.var().sort_values(ascending=False)
        X = X[var.index[:MAX_FEATURES]]
        notes.append(f"Kept the {MAX_FEATURES} highest-variance features.")
    if min(y.sum(), len(y) - y.sum()) < 10:
        raise DataValidationError("Each class needs at least 10 rows.")

    return Dataset(
        key="uploaded",
        name=name,
        X=X.reset_index(drop=True).astype(float),
        y=y,
        positive_label=positive_value,
        negative_label=neg_label,
        description=f"User-uploaded data: {len(X):,} rows, {X.shape[1]} features.",
        source="Uploaded CSV",
        notes=notes,
    )
