from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegularizerSpec:
    name: str
    supports_weight: bool
    notes: str


REGULARIZERS: dict[str, RegularizerSpec] = {
    "none": RegularizerSpec(
        name="none",
        supports_weight=False,
        notes="No auxiliary regularization beyond the main metric loss.",
    ),
    "embedding_norm": RegularizerSpec(
        name="embedding_norm",
        supports_weight=True,
        notes="Controls embedding magnitude or hypersphere concentration.",
    ),
    "language_alignment": RegularizerSpec(
        name="language_alignment",
        supports_weight=True,
        notes="Auxiliary text/metadata guidance when labels are weak or coarse.",
    ),
    "center_smoothing": RegularizerSpec(
        name="center_smoothing",
        supports_weight=True,
        notes="Encourages smoother class-center geometry across batches.",
    ),
}


def list_regularizers() -> list[str]:
    return sorted(REGULARIZERS)
