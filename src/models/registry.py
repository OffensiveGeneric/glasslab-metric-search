from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    embedding_dim: int
    supports_freeze_backbone: bool
    notes: str


BACKBONES: dict[str, BackboneSpec] = {
    "vit_base_patch16": BackboneSpec(
        name="vit_base_patch16",
        family="vision_transformer",
        embedding_dim=768,
        supports_freeze_backbone=True,
        notes="Default v0 baseline for art-metric search.",
    ),
    "convnext_base": BackboneSpec(
        name="convnext_base",
        family="convnext",
        embedding_dim=1024,
        supports_freeze_backbone=True,
        notes="Alternative convolutional baseline with stronger inductive bias.",
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        family="resnet",
        embedding_dim=2048,
        supports_freeze_backbone=True,
        notes="Lower-risk baseline for simpler ablations and bring-up.",
    ),
}


def list_backbones() -> list[str]:
    return sorted(BACKBONES)

