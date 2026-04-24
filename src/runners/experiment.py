from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from search.run_spec import RunSpec


def _score_from_payload(payload: dict[str, Any], salt: str) -> float:
    digest = hashlib.sha256((salt + json.dumps(payload, sort_keys=True)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def simulate_experiment(run_spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_spec.config

    retrieval = 0.45 + (0.40 * _score_from_payload(cfg, "retrieval"))
    verification = 0.50 + (0.35 * _score_from_payload(cfg, "verification"))
    robustness = 0.40 + (0.30 * _score_from_payload(cfg, "robustness"))
    instability_penalty = 0.05 * _score_from_payload(cfg, "penalty")

    metrics = {
        "run_id": run_spec.run_id,
        "dataset_id": run_spec.dataset.dataset_id,
        "retrieval_recall_at_10": round(min(retrieval, 0.99), 4),
        "forgery_auroc": round(min(verification, 0.99), 4),
        "robustness_score": round(min(robustness, 0.99), 4),
        "instability_penalty": round(instability_penalty, 4),
    }
    metrics["composite_score"] = round(
        (
            metrics["retrieval_recall_at_10"]
            + metrics["forgery_auroc"]
            + metrics["robustness_score"]
            - metrics["instability_penalty"]
        )
        / 3.0,
        4,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def simulate_contrastive_experiment(run_spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_spec.config
    
    dataset_id = run_spec.dataset.dataset_id
    
    grouped_rk = 0.55 + (0.35 * _score_from_payload(cfg, "grouped_rk"))
    opis = 0.35 + (0.25 * _score_from_payload(cfg, "opis"))
    ami = 0.50 + (0.30 * _score_from_payload(cfg, "ami"))
    ari = 0.48 + (0.28 * _score_from_payload(cfg, "ari"))
    nmi = 0.52 + (0.32 * _score_from_payload(cfg, "nmi"))
    silh = 0.45 + (0.35 * _score_from_payload(cfg, "silh"))
    
    metrics = {
        "run_id": run_spec.run_id,
        "dataset_id": dataset_id,
        "grouped_recall_at_k": round(min(grouped_rk, 0.99), 4),
        "opis": round(min(opis, 1.0), 4),
        "adjusted_mutual_info": round(min(ami, 1.0), 4),
        "adjusted_rand_index": round(min(ari, 1.0), 4),
        "normalized_mutual_info": round(min(nmi, 1.0), 4),
        "silhouette_score": round(min(silh, 1.0), 4),
    }
    
    metrics["composite_score"] = round(
        (
            metrics["grouped_recall_at_k"]
            + (1.0 - metrics["opis"])
            + metrics["adjusted_mutual_info"]
            + metrics["adjusted_rand_index"]
            + metrics["normalized_mutual_info"]
            + metrics["silhouette_score"]
        )
        / 6.0,
        4,
    )
    
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
    return metrics
