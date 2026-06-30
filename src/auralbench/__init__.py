"""Auditable evaluation for generated audio and speech models, with manifests, metrics, statistics, and reports."""

from .runner import evaluate_dataset, report_csv, report_html, report_json
from .text import character_error_rate, word_error_rate

__version__ = "0.1.2"
__all__ = [
    "__version__",
    "evaluate_dataset",
    "report_csv",
    "report_html",
    "report_json",
    "character_error_rate",
    "word_error_rate",
]
