#!/usr/bin/env python3
"""Debug script to diagnose baseline issues"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import get_dataloaders

# Load config
config = Config.from_yaml("configs/search_spaces/cifar100_smoke_test.yaml")

# Get dataloaders
dataloaders = get_dataloaders(config)

# Check dataloader sizes
print("=== Dataloader sizes ===")
for key, loader in dataloaders.items():
    if hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__'):
        print(f"{key}: {len(loader.dataset)} samples")

# Check class counts
print("\n=== Class counts ===")
for key in ["test_seen_0", "test_unseen_0"]:
    loader = dataloaders.get(key)
    if loader:
        labels = loader.dataset.dataset.targets
        indices = loader.dataset.indices
        class_counts = {}
        for idx in indices:
            label = labels[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"{key}: {len(class_counts)} classes, samples={len(indices)}")

print("\n=== Dataloader iteration test ===")
for split_name in ["test_seen_0", "test_unseen_0"]:
    loader = dataloaders.get(split_name)
    if loader:
        for i, (images, labels) in enumerate(loader):
            print(f"{split_name} batch {i}: images.shape={images.shape}, labels.shape={labels.shape}, unique_labels={torch.unique(labels)}")
            if i >= 1:
                break
