#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import get_dataloaders
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss
from src.models.backbone import ModelFactory
from src.metrics.metrics import AdvancedMetrics


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    config = Config()
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)

        if isinstance(loss_fn, SupervisedContrastiveLoss):
            loss = loss_fn(embeddings, labels)
        elif isinstance(loss_fn, TripletLoss):
            batch_size = embeddings.shape[0]
            anchor = embeddings
            positive = torch.zeros_like(embeddings)
            negative = torch.zeros_like(embeddings)

            for i in range(batch_size):
                same_class = torch.where(labels == labels[i])[0]
                diff_class = torch.where(labels != labels[i])[0]

                if len(same_class) > 1:
                    same_class = same_class[same_class != i]
                    positive[i] = embeddings[same_class[0]]
                else:
                    positive[i] = embeddings[i]

                if len(diff_class) > 0:
                    negative[i] = embeddings[diff_class[0]]
                else:
                    negative[i] = embeddings[(i + 1) % batch_size]

            loss = loss_fn(anchor, positive, negative)
        else:
            loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if batch_idx > 10:
            break

    return total_loss / max(num_batches, 1)


def run_pipeline(config_path: str, output_dir: Path):
    config = load_config(config_path)
    device = config.training.device
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train_seen_0"]

    backbone_name = config.model.backbones[0] if config.model.backbones else "resnet18"
    model = ModelFactory.create_backbone(config, backbone_name)
    model = model.to(device)

    loss_config = config.loss.contrastive if hasattr(config.loss.contrastive, "get") else {}
    loss_fn = SupervisedContrastiveLoss(temperature=loss_config.get("temperature", 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    num_epochs = min(config.training.epochs, 2)
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_pipeline(args.config, Path(args.output_dir))
