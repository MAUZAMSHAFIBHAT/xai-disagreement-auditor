"""Central configuration: method names, defaults and colour tokens."""

from __future__ import annotations

import os

# Explanation methods, in a fixed order. The order is also the colour order
# used in every chart, so a method always keeps the same colour.
METHODS: list[str] = ["shapley", "lime", "integrated_gradients", "smoothgrad_x_input", "occlusion"]

METHOD_LABELS: dict[str, str] = {
    "shapley": "Shapley (sampling)",
    "lime": "LIME",
    "integrated_gradients": "Integrated Gradients",
    "smoothgrad_x_input": "SmoothGrad × Input",
    "occlusion": "Occlusion",
}

METHOD_SHORT: dict[str, str] = {
    "shapley": "SHAP",
    "lime": "LIME",
    "integrated_gradients": "IG",
    "smoothgrad_x_input": "SG×I",
    "occlusion": "OCC",
}

METHOD_FAMILY: dict[str, str] = {
    "shapley": "Game theory (perturbation)",
    "lime": "Local surrogate (perturbation)",
    "integrated_gradients": "Path integral (gradient)",
    "smoothgrad_x_input": "Local gradient",
    "occlusion": "Ablation (perturbation)",
}

# Validated categorical palette (light surface), slots 1-5, plus a neutral for "Random".
METHOD_COLORS: dict[str, str] = {
    "shapley": "#2a78d6",
    "lime": "#eb6834",
    "integrated_gradients": "#1baf7a",
    "smoothgrad_x_input": "#eda100",
    "occlusion": "#e87ba4",
    "random": "#898781",
}

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "Neural Network (MLP)",
}

# Trust score bands (0-100). Calibrated on the bundled datasets; see docs/METHODOLOGY.md.
TRUST_HIGH = 70.0
TRUST_LOW = 50.0

DEFAULT_TOP_K = 5
DEFAULT_SEED = 42

# Budget knobs for each explainer. Higher = slower but more stable.
SHAPLEY_PERMUTATIONS = int(os.getenv("XAUDIT_SHAPLEY_PERMUTATIONS", "256"))
LIME_SAMPLES = int(os.getenv("XAUDIT_LIME_SAMPLES", "1500"))
IG_STEPS = int(os.getenv("XAUDIT_IG_STEPS", "48"))
SMOOTHGRAD_SAMPLES = int(os.getenv("XAUDIT_SMOOTHGRAD_SAMPLES", "24"))
BACKGROUND_SIZE = int(os.getenv("XAUDIT_BACKGROUND_SIZE", "100"))
MAX_UPLOAD_ROWS = int(os.getenv("XAUDIT_MAX_UPLOAD_ROWS", "20000"))
MAX_FEATURES = int(os.getenv("XAUDIT_MAX_FEATURES", "60"))
