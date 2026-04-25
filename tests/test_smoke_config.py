from __future__ import annotations

import pytest
from pathlib import Path

from src.config import Config


def test_config_loads():
    """Smoke test: config can be loaded from YAML"""
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    config = Config.from_yaml(str(config_path))
    assert config is not None
    assert hasattr(config, "data")
    assert hasattr(config, "model")
