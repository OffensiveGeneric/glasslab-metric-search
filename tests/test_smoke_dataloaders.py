from __future__ import annotations

import pytest
from pathlib import Path

from src.config import Config
from src.data.dataset import get_dataloaders


def test_dataloader_splits():
    """Smoke test: dataloaders split into train_seen/val_seen/test_seen"""
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    config = Config.from_yaml(str(config_path))
    dataloaders = get_dataloaders(config)
    
    assert "train_seen_0" in dataloaders
    assert "val_seen_0" in dataloaders
    assert "test_seen_0" in dataloaders
    
    train_loader = dataloaders["train_seen_0"]
    val_loader = dataloaders["val_seen_0"]
    test_loader = dataloaders["test_seen_0"]
    
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    
    assert len(train_loader.dataset) > len(val_loader.dataset)
    assert len(train_loader.dataset) > len(test_loader.dataset)
