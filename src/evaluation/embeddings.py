"""
Embedding collection utilities: Shared between run_experiment.py and run_baseline.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


def compute_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config,
    warnings: list[str],
    context: str,
) -> Dict[str, Any]:
    """Compute all metrics for embeddings
    
    Args:
        embeddings: Embedding tensor
        labels: Label tensor
        config: Experiment config
        warnings: List to append warnings to
        context: Context string for warnings
        
    Returns:
        Dictionary of metrics
    """
    from src.runners.trainer import evaluate_embeddings
    
    return evaluate_embeddings(embeddings, labels, config, warnings, context)
