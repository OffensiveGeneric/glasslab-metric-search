from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorSpec:
    name: str
    metrics: tuple[str, ...]
    notes: str


EVALUATORS: dict[str, EvaluatorSpec] = {
    "cifar_contrastive_v1": EvaluatorSpec(
        name="cifar_contrastive_v1",
        metrics=(
            "grouped_recall_at_k",
            "opis",
            "ami",
            "ari",
            "normalized_mutual_information",
            "silhouette_score",
        ),
        notes="Contrastive learning evaluator with unseen class generalization metrics.",
    )
}


def list_evaluators() -> list[str]:
    return sorted(EVALUATORS)
