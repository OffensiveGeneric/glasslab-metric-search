from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RankedResult:
    run_id: str
    composite_score: float
    metrics: dict
    path: Path


def composite_score(metrics: dict) -> float:
    retrieval = float(metrics.get("retrieval_recall_at_10", 0.0))
    verification = float(metrics.get("forgery_auroc", 0.0))
    robustness = float(metrics.get("robustness_score", 0.0))
    penalty = float(metrics.get("instability_penalty", 0.0))
    return (0.45 * retrieval) + (0.35 * verification) + (0.20 * robustness) - penalty


def load_run_results(root: Path) -> list[RankedResult]:
    results: list[RankedResult] = []
    for metrics_path in root.glob("**/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append(
            RankedResult(
                run_id=str(payload["run_id"]),
                composite_score=composite_score(payload),
                metrics=payload,
                path=metrics_path,
            )
        )
    return results


def rank_candidates(results: list[RankedResult]) -> list[RankedResult]:
    return sorted(results, key=lambda item: item.composite_score, reverse=True)

