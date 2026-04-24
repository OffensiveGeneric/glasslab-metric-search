from __future__ import annotations

from search.selection import composite_score


def summarize_candidate(metrics: dict) -> dict[str, float]:
    return {
        "retrieval_recall_at_10": float(metrics["retrieval_recall_at_10"]),
        "forgery_auroc": float(metrics["forgery_auroc"]),
        "robustness_score": float(metrics["robustness_score"]),
        "composite_score": composite_score(metrics),
    }


def summarize_contrastive_candidate(metrics: dict) -> dict[str, float]:
    return {
        "grouped_recall_at_k": float(metrics["grouped_recall_at_k"]),
        "opis": float(metrics["opis"]),
        "adjusted_mutual_info": float(metrics["adjusted_mutual_info"]),
        "adjusted_rand_index": float(metrics["adjusted_rand_index"]),
        "normalized_mutual_info": float(metrics["normalized_mutual_info"]),
        "silhouette_score": float(metrics["silhouette_score"]),
        "composite_score": float(metrics["composite_score"]),
    }

