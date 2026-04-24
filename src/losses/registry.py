from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossSpec:
    name: str
    family: str
    supports_margin: bool
    supports_temperature: bool
    notes: str


LOSSES: dict[str, LossSpec] = {
    "contrastive": LossSpec(
        name="contrastive",
        family="pair",
        supports_margin=True,
        supports_temperature=False,
        notes="Basic pairwise baseline.",
    ),
    "triplet": LossSpec(
        name="triplet",
        family="pair",
        supports_margin=True,
        supports_temperature=False,
        notes="Triplet-family baseline with mining sensitivity.",
    ),
    "multi_similarity": LossSpec(
        name="multi_similarity",
        family="pair",
        supports_margin=True,
        supports_temperature=True,
        notes="Pair-based formulation with richer positive/negative interactions.",
    ),
    "proxy_anchor": LossSpec(
        name="proxy_anchor",
        family="proxy",
        supports_margin=True,
        supports_temperature=True,
        notes="Strong proxy-based baseline for large-batch efficiency.",
    ),
    "proxy_nca": LossSpec(
        name="proxy_nca",
        family="proxy",
        supports_margin=False,
        supports_temperature=True,
        notes="Proxy-based objective with simpler class-representative structure.",
    ),
    "proxy_gml": LossSpec(
        name="proxy_gml",
        family="proxy",
        supports_margin=True,
        supports_temperature=True,
        notes="Proxy-graph style placeholder family for richer class structure.",
    ),
}


def list_losses() -> list[str]:
    return sorted(LOSSES)
