"""
Dataset resolver: Handle multiple dataset sources (torchvision, S3, etc.)

This module provides a unified interface for loading datasets from different sources:
- torchvision (CIFAR-100, etc.)
- Object store (S3, GCS)
- Local files (parquet, pickle, etc.)

Example usage:
    from src.data.resolver import resolve_dataset, TorchvisionSource
    
    # Load CIFAR-100 from torchvision
    source = TorchvisionSource(name="cifar100", root="/mnt/datasets/cifar100")
    train_dataset = source.load(train=True)
    
    # Or use config-based resolution
    config = {"type": "torchvision", "name": "cifar100", "root": "/mnt/datasets/cifar100"}
    source = resolve_dataset(config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DatasetSource(Protocol):
    """Interface for dataset sources"""
    
    def load(self, train: bool) -> object:
        """Load dataset (returns torch.utils.data.Dataset compatible object)"""
        ...


@dataclass
class TorchvisionSource:
    """Load datasets from torchvision (e.g., CIFAR-100)"""
    name: str
    root: str = "./data"
    download: bool = True
    
    def load(self, train: bool, transform=None) -> object:
        if self.name == "cifar100":
            from torchvision import datasets
            return datasets.CIFAR100(
                root=self.root,
                train=train,
                download=self.download,
                transform=transform
            )
        raise ValueError(f"Unknown torchvision dataset: {self.name}")


@dataclass
class ObjectStoreSource:
    """Load datasets from object store (e.g., S3)"""
    train_uri: str
    val_uri: str | None = None
    test_uri: str | None = None
    
    def load(self, train: bool) -> object:
        # Placeholder for S3/object store loading
        # Would use pandas.parquet, s3fs, etc.
        raise NotImplementedError("Object store loading not yet implemented")


def resolve_dataset(source_config: dict) -> DatasetSource:
    """Create appropriate DatasetSource based on config"""
    source_type = source_config.get("type", "torchvision")
    
    if source_type == "torchvision":
        return TorchvisionSource(
            name=source_config.get("name", "cifar100"),
            root=source_config.get("root", "./data"),
            download=source_config.get("download", True)
        )
    elif source_type == "object_store":
        return ObjectStoreSource(
            train_uri=source_config["train_uri"],
            val_uri=source_config.get("val_uri"),
            test_uri=source_config.get("test_uri")
        )
    else:
        raise ValueError(f"Unknown dataset source type: {source_type}")
