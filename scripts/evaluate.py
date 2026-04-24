#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.art_retrieval import summarize_candidate
from src.metrics.cifar_contrastive import (
    compute_ami,
    compute_ari,
    compute_nmi,
    compute_opis,
    compute_silhouette,
    grouped_recall_at_k,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--mode", default="contrastive", choices=["standard", "contrastive"])
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    
    if args.mode == "contrastive":
        print(json.dumps(summarize_contrastive_candidate(metrics), indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize_candidate(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
