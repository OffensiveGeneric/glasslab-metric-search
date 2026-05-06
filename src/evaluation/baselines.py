"""
Baseline utilities: Random and shuffled-label baselines
"""

from __future__ import annotations

import torch
from typing import Tuple, Dict, Any


def compute_random_baseline(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config,
    warnings: list[str],
    context: str,
) -> Dict[str, Any]:
    """Compute random embedding baseline
    
    Args:
        embeddings: Original embeddings
        labels: Labels tensor
        config: Experiment config
        warnings: List to append warnings to
        context: Context string for warnings
        
    Returns:
        Dictionary of random baseline metrics
    """
    from src.runners.trainer import random_embedding_baseline
    
    return random_embedding_baseline(embeddings, labels, config, warnings, context)


def compute_shuffled_baseline(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config,
    warnings: list[str],
    context: str,
) -> Dict[str, Any]:
    """Compute shuffled label baseline
    
    Args:
        embeddings: Embeddings tensor
        labels: Original labels tensor
        config: Experiment config
        warnings: List to append warnings to
        context: Context string for warnings
        
    Returns:
        Dictionary of shuffled baseline metrics
    """
    from src.runners.trainer import shuffled_label_baseline
    
    return shuffled_label_baseline(embeddings, labels, config, warnings, context)
