"""
Evaluation module: Shared evaluation logic for training and baselines
"""

from src.evaluation.embeddings import collect_embeddings, evaluate_embeddings, random_embedding_baseline, shuffled_label_baseline
from src.evaluation.reports import generate_report, serialize_metrics

__all__ = [
    "collect_embeddings",
    "evaluate_embeddings",
    "random_embedding_baseline",
    "shuffled_label_baseline",
    "generate_report",
    "serialize_metrics",
]
