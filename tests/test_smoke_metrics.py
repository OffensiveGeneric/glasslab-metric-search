from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.backbone import ModelFactory


def test_metrics_sanity():
    """Smoke test: metrics compute reasonable values"""
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    config = Config.from_yaml(str(config_path))
    dataloaders = get_dataloaders(config)
    
    train_loader = dataloaders["train_seen_0"]
    images, labels = next(iter(train_loader))
    
    device = config.training.device
    model = ModelFactory.create_backbone(config, "resnet18")
    model = model.to(device)
    
    images = images.to(device)
    labels = labels.to(device)
    
    embeddings = model(images)
    embeddings = embeddings.detach()
    
    # Basic metrics
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    nmi = normalized_mutual_info_score(labels_np, labels_np)
    ami = adjusted_mutual_info_score(labels_np, labels_np)
    ari = adjusted_rand_score(labels_np, labels_np)
    silh = silhouette_score(embeddings_np, labels_np)
    
    assert 0.0 <= nmi <= 1.0
    assert 0.0 <= ami <= 1.0
    assert 0.0 <= ari <= 1.0
    assert -1.0 <= silh <= 1.0
    
    assert silh < 0.99, "Silhouette should not be perfect (implies incorrect metric computation)"
