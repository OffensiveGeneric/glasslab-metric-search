from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinerSpec:
    name: str
    family: str
    notes: str


MINERS: dict[str, MinerSpec] = {
    "batch_hard": MinerSpec(
        name="batch_hard",
        family="pair",
        notes="Uses hardest positives/negatives inside the batch.",
    ),
    "semi_hard": MinerSpec(
        name="semi_hard",
        family="pair",
        notes="Prefers margin-violating but non-degenerate negatives.",
    ),
    "distance_weighted": MinerSpec(
        name="distance_weighted",
        family="pair",
        notes="Biases selection toward informative medium-distance negatives.",
    ),
    "multi_similarity": MinerSpec(
        name="multi_similarity",
        family="pair",
        notes="Pairs naturally with multi-similarity style objectives.",
    ),
    "hard_negative": MinerSpec(
        name="hard_negative",
        family="pair",
        notes="Aggressive hard-negative mining, useful but less stable.",
    ),
}


def list_miners() -> list[str]:
    return sorted(MINERS)
