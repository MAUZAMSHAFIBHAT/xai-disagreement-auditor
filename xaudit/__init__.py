"""XAI Disagreement Auditor.

Measures how much popular post-hoc explanation methods agree with each other
on the same prediction, turns that into a per-prediction Explanation Trust
Score, and checks each method against ground truth and faithfulness tests.
"""

__version__ = "1.0.0"

from xaudit.config import METHODS, METHOD_LABELS  # noqa: E402,F401
