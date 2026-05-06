"""
Losses module: Metric learning loss functions and factory
"""

from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss, HardNegativeMiner
from src.losses.registry import LOSSES, LossSpec, list_losses
from src.losses.factory import (
    create_loss,
    get_loss_spec,
    list_supported_losses,
    check_loss_supported,
    check_loss_implementation,
)

__all__ = [
    "SupervisedContrastiveLoss",
    "TripletLoss",
    "ShadowLoss",
    "HardNegativeMiner",
    "LOSSES",
    "LossSpec",
    "list_losses",
    "create_loss",
    "get_loss_spec",
    "list_supported_losses",
    "check_loss_supported",
    "check_loss_implementation",
]
