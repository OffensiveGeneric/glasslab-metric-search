from __future__ import annotations

from src.evaluators.registry import EVALUATORS
from src.losses.registry import LOSSES
from src.miners.registry import MINERS
from src.models.registry import BACKBONES
from src.regularizers.registry import REGULARIZERS


def validate_experiment_config(config: dict) -> None:
    backbone = config.get("backbone", {}).get("name")
    loss = config.get("loss", {}).get("name")
    miner = config.get("miner", {}).get("name")
    regularizer = config.get("regularizer", {}).get("name")
    evaluator = config.get("evaluator", {}).get("name")

    checks = (
        ("backbone", backbone, BACKBONES),
        ("loss", loss, LOSSES),
        ("miner", miner, MINERS),
        ("regularizer", regularizer, REGULARIZERS),
        ("evaluator", evaluator, EVALUATORS),
    )

    for field, value, registry in checks:
        if not value:
            raise ValueError(f"experiment.{field}.name is required")
        if value not in registry:
            allowed = ", ".join(sorted(registry))
            raise ValueError(f"unsupported {field} '{value}'; allowed: {allowed}")


def validate_cifar100_config(config: dict) -> None:
    validate_experiment_config(config)
    
    dataset_id = config.get("dataset", {}).get("dataset_id", "")
    if "cifar100" in dataset_id.lower():
        # For contrastive learning pipeline
        pipeline = config.get("pipeline", "contrastive")
        
        if pipeline == "contrastive_learning":
            # Validate contrastive learning specific fields
            model_family = config.get("model_family", "")
            if not model_family:
                raise ValueError("contrastive learning requires model_family")
            
            loss_name = config.get("loss_name", "contrastive")
            if loss_name not in ["contrastive", "triplet", "multi_similarity", "proxy_anchor"]:
                allowed = ", ".join(["contrastive", "triplet", "multi_similarity", "proxy_anchor"])
                raise ValueError(f"CIFAR-100 contrastive learning requires loss_name from: {allowed}")
            
            backbone_name = config.get("backbone_name", "resnet50")
            if backbone_name not in ["resnet50", "vit_base_patch16", "convnext_base"]:
                allowed = ", ".join(["resnet50", "vit_base_patch16", "convnext_base"])
                raise ValueError(f"CIFAR-100 contrastive learning requires backbone_name from: {allowed}")
        else:
            # Legacy validation
            backbone = config.get("backbone", {}).get("name", "")
            if backbone not in ["resnet50", "vit_base_patch16", "convnext_base"]:
                allowed = ", ".join(["resnet50", "vit_base_patch16", "convnext_base"])
                raise ValueError(f"CIFAR-100 experiments require backbone from: {allowed}")
            
            loss = config.get("loss", {}).get("name", "")
            if loss not in ["contrastive", "triplet", "multi_similarity", "proxy_anchor"]:
                allowed = ", ".join(["contrastive", "triplet", "multi_similarity", "proxy_anchor"])
                raise ValueError(f"CIFAR-100 experiments require loss from: {allowed}")
            
            miner = config.get("miner", {}).get("name", "")
            if miner not in ["batch_hard", "semi_hard", "distance_weighted", "multi_similarity", "hard_negative"]:
                allowed = ", ".join(["batch_hard", "semi_hard", "distance_weighted", "multi_similarity", "hard_negative"])
                raise ValueError(f"CIFAR-100 experiments require miner from: {allowed}")
