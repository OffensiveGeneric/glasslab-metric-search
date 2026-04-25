from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.backbone import ModelFactory
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss


def test_loss_forward():
    """Smoke test: losses compute without error"""
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
    
    # Test SupervisedContrastiveLoss
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    loss = loss_fn(embeddings, labels)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test TripletLoss
    loss_fn = TripletLoss(margin=0.3)
    batch_size = embeddings.shape[0]
    anchor = embeddings
    positive = torch.roll(embeddings, shifts=1, dims=0)
    negative = torch.roll(embeddings, shifts=-1, dims=0)
    loss = loss_fn(anchor, positive, negative)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test ShadowLoss
    loss_fn = ShadowLoss(embedding_dim=config.model.embedding_dim)
    loss = loss_fn(anchor, positive, negative)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
