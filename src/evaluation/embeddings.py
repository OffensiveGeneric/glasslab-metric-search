"""
Embedding collection utilities: Shared between run_experiment.py and run_baseline.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def random_embedding_baseline(
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
    generator = torch.Generator().manual_seed(int(config.data.seed) + 17)
    random_embeddings = torch.randn(
        embeddings.shape,
        generator=generator,
        dtype=embeddings.dtype if embeddings.numel() else torch.float32,
    )
    metrics = evaluate_embeddings(random_embeddings, labels, config, warnings, context)
    
    random_global_recall = metrics.get("global_recall_at_1")
    random_global_chance_exact = metrics.get("global_recall_at_1_chance_exact")
    random_global_chance_approx = metrics.get("global_recall_at_1_chance_approx")
    random_grouped_recall = metrics.get("grouped_recall_at_k")
    random_grouped_chance = metrics.get("grouped_recall_chance_at_k")
    
    metrics["global_recall_at_1_expected_target"] = random_global_chance_exact
    metrics["global_recall_at_1_abs_error"] = (
        None
        if random_global_recall is None or random_global_chance_exact is None
        else round(float(abs(random_global_recall - random_global_chance_exact)), 4)
    )
    metrics["global_recall_at_1_sanity_pass"] = (
        False
        if metrics["global_recall_at_1_abs_error"] is None
        else metrics["global_recall_at_1_abs_error"] <= 0.03
    )
    metrics["grouped_recall_at_k_expected_target"] = random_grouped_chance
    metrics["grouped_recall_at_k_abs_error"] = (
        None
        if random_grouped_recall is None or random_grouped_chance is None
        else round(float(abs(random_grouped_recall - random_grouped_chance)), 4)
    )
    metrics["global_recall_at_1_expected_target_approx"] = random_global_chance_approx
    metrics["global_recall_at_1_chance_approx"] = random_global_chance_approx
    return metrics


def shuffled_label_baseline(
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
    generator = torch.Generator().manual_seed(int(config.data.seed))
    shuffled = labels[torch.randperm(labels.numel(), generator=generator)]
    return evaluate_embeddings(embeddings, shuffled, config, warnings, context)


def baseline_sane(
    observed: Any,
    chance: Any,
    *,
    absolute_tolerance: float = 0.10,
    relative_tolerance: float = 0.50,
) -> bool:
    """Check if baseline metrics are reasonable
    
    Args:
        observed: Observed metric value
        chance: Expected chance value
        absolute_tolerance: Absolute tolerance
        relative_tolerance: Relative tolerance
        
    Returns:
        True if baseline is sane (near chance level)
    """
    if observed is None or chance is None:
        return False
    observed_f = float(observed)
    chance_f = float(chance)
    tolerance = max(absolute_tolerance, abs(chance_f) * relative_tolerance)
    return abs(observed_f - chance_f) <= tolerance


def evaluate_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config,
    warnings: list[str],
    context: str,
) -> Dict[str, Any]:
    """Compute metrics for embeddings
    
    Args:
        embeddings: Embedding tensor
        labels: Label tensor
        config: Experiment config
        warnings: List to append warnings to
        context: Context string for warnings
        
    Returns:
        Dictionary of metrics
    """
    if embeddings.numel() == 0 or labels.numel() == 0:
        warnings.append(f"{context} has no embeddings or labels; metrics are null")
        return {
            "grouped_recall_at_k": None,
            "opis": None,
            "adjusted_mutual_info": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None,
            "silhouette_score": None,
        }

    num_classes = int(torch.unique(labels.detach().cpu()).numel())
    if num_classes < 2:
        warnings.append(f"{context} has fewer than two classes; metrics are null")
        return {
            "grouped_recall_at_k": None,
            "opis": None,
            "adjusted_mutual_info": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None,
            "silhouette_score": None,
        }

    try:
        from src.metrics.metrics import AdvancedMetrics
        return AdvancedMetrics(config).compute_all_metrics(embeddings, labels)
    except Exception as exc:
        warnings.append(f"{context} metrics could not be computed: {exc}")
        return {
            "grouped_recall_at_k": None,
            "opis": None,
            "adjusted_mutual_info": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None,
            "silhouette_score": None,
        }


def collect_embeddings(
    model: nn.Module,
    loader: DataLoader | None,
    device: str,
    max_batches: int | None,
    db_path: Path | None = None,
    table_name: str = "embeddings",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect embeddings from model with optional SQLite storage
    
    Args:
        model: Trained model
        loader: Data loader
        device: Device to run inference on
        max_batches: Maximum batches to process
        db_path: Optional SQLite path for streaming embeddings to disk
        table_name: SQLite table name
        
    Returns:
        Tuple of (embeddings_tensor, labels_tensor)
    """
    if loader is None:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    model.eval()
    all_embeddings = []
    all_labels = []
    
    conn = None
    cursor = None
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                split_name TEXT,
                batch_idx INTEGER,
                embeddings BLOB,
                labels BLOB,
                PRIMARY KEY (split_name, batch_idx)
            )
        """)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            if cursor:
                emb_cpu = embeddings.cpu()
                embeddings_np = emb_cpu.numpy()
                labels_np = labels.cpu().numpy()
                cursor.execute(
                    f"INSERT OR REPLACE INTO {table_name} (split_name, batch_idx, embeddings, labels) VALUES (?, ?, ?, ?)",
                    ("main", batch_idx, embeddings_np.tobytes(), labels_np.tobytes())
                )
                conn.commit()
    
    if conn:
        conn.close()
    
    if not all_embeddings:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings), torch.cat(all_labels)



