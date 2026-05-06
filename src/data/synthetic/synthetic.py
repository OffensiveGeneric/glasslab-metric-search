"""
Synthetic dataset: Tiny dataset for local smoke tests without internet
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Literal


class SyntheticDataset(Dataset):
    """Synthetic dataset for local smoke testing
    
    Creates a tiny dataset with few samples per class to test
    the full pipeline without needing internet or large downloads.
    
    Args:
        split: Which split to create ("train", "val", "test")
        num_classes: Number of classes (default 10 for quick tests)
        samples_per_class: Number of samples per class (default 2)
        embedding_dim: Dimension of synthetic features (default 64)
        noise_level: Noise level in features (default 0.1)
    """
    
    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        num_classes: int = 10,
        samples_per_class: int = 2,
        embedding_dim: int = 64,
        noise_level: float = 0.1,
    ):
        self.split = split
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.embedding_dim = embedding_dim
        self.noise_level = noise_level
        
        # Generate class centers
        self.class_centers = torch.randn(num_classes, embedding_dim)
        
        # Generate samples
        self.data = []
        self.labels = []
        
        for class_idx in range(num_classes):
            center = self.class_centers[class_idx]
            for _ in range(samples_per_class):
                # Add noise to class center
                sample = center + torch.randn(embedding_dim) * noise_level
                self.data.append(sample)
                self.labels.append(class_idx)
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        
        print(f"SyntheticDataset[{split}]: {len(self)} samples, {num_classes} classes")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def create_synthetic_dataloaders(
    batch_size: int = 4,
    num_classes: int = 10,
    samples_per_class: int = 2,
    embedding_dim: int = 64,
    num_workers: int = 0,
) -> dict:
    """Create synthetic dataloaders for smoke testing
    
    Args:
        batch_size: Batch size
        num_classes: Number of classes
        samples_per_class: Samples per class
        embedding_dim: Feature dimension
        num_workers: Number of DataLoader workers
        
    Returns:
        Dictionary of dataloaders
    """
    train_dataset = SyntheticDataset(
        split="train",
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        embedding_dim=embedding_dim,
    )
    
    val_dataset = SyntheticDataset(
        split="val",
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        embedding_dim=embedding_dim,
    )
    
    test_dataset = SyntheticDataset(
        split="test",
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        embedding_dim=embedding_dim,
    )
    
    return {
        "train_seen_0": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val_seen_0": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test_seen_0": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test_unseen_0": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
