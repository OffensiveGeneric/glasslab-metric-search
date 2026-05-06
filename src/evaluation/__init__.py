"""
Evaluation module: Shared evaluation logic for training and baselines
"""

from src.evaluation.embeddings import collect_embeddings, compute_metrics
from src.evaluation.baselines import compute_random_baseline, compute_shuffled_baseline
from src.evaluation.reports import generate_report, serialize_metrics

__all__ = [
    "collect_embeddings",
    "compute_metrics",
    "compute_random_baseline",
    "compute_shuffled_baseline",
    "generate_report",
    "serialize_metrics",
]
