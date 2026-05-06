"""
Loss factory: Construct loss functions based on registry and config
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from src.losses.registry import LOSSES, LossSpec
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss


def create_loss(loss_name: str, config: dict | None = None) -> nn.Module:
    """Create loss function based on name and config
    
    Args:
        loss_name: Name of loss function (e.g., "contrastive", "triplet", "multi_similarity")
        config: Loss-specific configuration dict
        
    Returns:
        nn.Module: Initialized loss function
        
    Raises:
        ValueError: If loss_name not in registry
        NotImplementedError: If loss not yet implemented
    """
    if loss_name not in LOSSES:
        raise ValueError(f"Unknown loss '{loss_name}'. Supported losses: {list(LOSSES.keys())}")
    
    spec = LOSSES[loss_name]
    config = config or {}
    
    if loss_name == "contrastive":
        return SupervisedContrastiveLoss(
            temperature=config.get("temperature", 0.1),
            base_temperature=config.get("base_temperature", 0.1)
        )
    
    elif loss_name == "triplet":
        return TripletLoss(
            margin=config.get("margin", 0.3),
            miner=config.get("miner", "semi_hard")
        )
    
    elif loss_name == "shadow":
        return ShadowLoss(
            embedding_dim=config.get("embedding_dim", 512),
            projection_dim=config.get("projection_dim", 1),
            learnable_projection=config.get("learnable_projection", True)
        )
    
    elif loss_name == "multi_similarity":
        raise NotImplementedError("Multi-similarity loss not yet implemented")
    
    elif loss_name == "proxy_anchor":
        raise NotImplementedError("Proxy-anchor loss not yet implemented")
    
    elif loss_name == "proxy_nca":
        raise NotImplementedError("Proxy-NCA loss not yet implemented")
    
    elif loss_name == "proxy_gml":
        raise NotImplementedError("Proxy-GML loss not yet implemented")
    
    else:
        raise NotImplementedError(f"Loss '{loss_name}' not implemented")


def get_loss_spec(loss_name: str) -> LossSpec:
    """Get loss specification from registry"""
    if loss_name not in LOSSES:
        raise ValueError(f"Unknown loss '{loss_name}'. Supported losses: {list(LOSSES.keys())}")
    return LOSSES[loss_name]


def list_supported_losses() -> list[str]:
    """List all supported loss functions"""
    return list(LOSSES.keys())


def check_loss_supported(loss_name: str) -> bool:
    """Check if loss is supported (in registry)"""
    return loss_name in LOSSES


def check_loss_implementation(loss_name: str) -> bool:
    """Check if loss is fully implemented (not just in registry)"""
    if loss_name not in LOSSES:
        return False
    
    try:
        create_loss(loss_name, {})
        return True
    except NotImplementedError:
        return False
