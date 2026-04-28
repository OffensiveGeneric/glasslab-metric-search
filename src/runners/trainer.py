from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from search.run_spec import RunSpec
from src.config import Config
from src.data.dataset import get_dataloaders
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss
from src.models.backbone import ModelFactory
from src.metrics.metrics import AdvancedMetrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    max_batches: int = 100,
) -> float:
    """Train for one epoch with early stopping for speed."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

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

    return total_loss / max(num_batches, 1)


def evaluate_metrics(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: str,
    config: Config,
) -> Dict[str, float]:
    """Evaluate on test set and compute metrics."""
    model.eval()
    all_embeddings = []
    all_labels = []

    test_loader = dataloaders.get("test_seen_0")
    if test_loader is None:
        return {}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    metrics_fn = AdvancedMetrics(config)
    return metrics_fn.compute_all_metrics(embeddings, labels)


def run_real_experiment(run_spec: RunSpec, output_dir: Path) -> Dict[str, Any]:
    """Run a real training experiment on CIFAR-100."""
    print(f"Starting run_real_experiment", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml")

    if run_spec.config:
        from src.config import DataConfig, AugmentationConfig, LossConfig, ModelConfig, TrainingConfig, HPOConfig, EvaluationConfig, L2ANCConfig
        
        for key, value in run_spec.config.items():
            if key == "data":
                if isinstance(config.data, DataConfig):
                    config.data = DataConfig(**value)
                else:
                    config.data = DataConfig(**value)
            elif key == "augmentation":
                if isinstance(config.augmentation, AugmentationConfig):
                    config.augmentation = AugmentationConfig(**value)
                else:
                    config.augmentation = AugmentationConfig(**value)
            elif key == "loss":
                if isinstance(config.loss, LossConfig):
                    config.loss = LossConfig(**value)
                else:
                    config.loss = LossConfig(**value)
            elif key == "model":
                if isinstance(config.model, ModelConfig):
                    config.model = ModelConfig(**value)
                else:
                    config.model = ModelConfig(**value)
            elif key == "training":
                if isinstance(config.training, TrainingConfig):
                    config.training = TrainingConfig(**value)
                else:
                    config.training = TrainingConfig(**value)
            elif key == "hpo":
                if isinstance(config.hpo, HPOConfig):
                    config.hpo = HPOConfig(**value)
                else:
                    config.hpo = HPOConfig(**value)
            elif key == "evaluation":
                if isinstance(config.evaluation, EvaluationConfig):
                    config.evaluation = EvaluationConfig(**value)
                else:
                    config.evaluation = EvaluationConfig(**value)
            elif key == "l2anc":
                if isinstance(config.l2anc, L2ANCConfig):
                    config.l2anc = L2ANCConfig(**value)
                else:
                    config.l2anc = L2ANCConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        flat_config_map = {
            "backbone_name": lambda v: setattr(config.model, "backbones", [v] if isinstance(v, str) else v),
            "loss_name": lambda v: None,
            "batch_size": lambda v: setattr(config.training, "batch_size", v),
            "learning_rate": lambda v: setattr(config.training, "learning_rate", v),
            "max_epochs": lambda v: setattr(config.training, "epochs", v),
            "temperature": lambda v: config.loss.contrastive.__setitem__("temperature", v) if isinstance(config.loss.contrastive, dict) else None,
        }
        
        for key, value in run_spec.config.items():
            if key not in ["data", "augmentation", "loss", "model", "training", "hpo", "evaluation", "l2anc"]:
                if key in flat_config_map:
                    flat_config_map[key](value)
        
        loss_name = run_spec.config.get("loss_name", "contrastive")
        if loss_name != "contrastive":
            raise NotImplementedError(f"Only contrastive loss is supported in real mode, got: {loss_name}")

    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train_seen_0"]

    device = config.training.device
    if not torch.cuda.is_available():
        device = "cpu"

    backbone_name = config.model.backbones[0] if config.model.backbones else "resnet18"
    model = ModelFactory.create_backbone(config, backbone_name)
    model = model.to(device)

    loss_config = config.loss.contrastive if hasattr(config.loss.contrastive, "get") else {}
    loss_fn = SupervisedContrastiveLoss(temperature=loss_config.get("temperature", 0.1))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    num_epochs = min(run_spec.budget.max_epochs if run_spec.budget.max_epochs is not None else config.training.epochs, 2)
    max_train_batches = run_spec.budget.max_train_batches if run_spec.budget.max_train_batches is not None else 100
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, max_train_batches)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    max_eval_batches = run_spec.budget.max_eval_batches if run_spec.budget.max_eval_batches is not None else 100

    # Save checkpoints
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoints_dir / "final_model.pt")
    
    # Save embeddings dir
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect embeddings once for all splits
    def collect_embeddings_for_loader(dataloader, name):
        if dataloader is None:
            return None, None
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_eval_batches:
                    break
                images = images.to(device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        if not all_embeddings:
            return None, None
        
        embeddings = torch.cat(all_embeddings)
        labels = torch.cat(all_labels)
        
        # Save embeddings
        torch.save(embeddings, embeddings_dir / f"{name}_embeddings.pt")
        torch.save(labels, embeddings_dir / f"{name}_labels.pt")
        
        return embeddings, labels
    
    val_seen_embeddings, val_seen_labels = collect_embeddings_for_loader(dataloaders.get("val_seen_0"), "val_seen")
    test_seen_embeddings, test_seen_labels = collect_embeddings_for_loader(dataloaders.get("test_seen_0"), "test_seen")
    test_unseen_embeddings, test_unseen_labels = collect_embeddings_for_loader(dataloaders.get("test_unseen_0"), "test_unseen")
    
    # Compute metrics from collected embeddings
    def compute_split_metrics_from_embeddings(embeddings, labels, split_name, config):
        if embeddings is None or labels is None:
            return {}
        
        metrics_fn = AdvancedMetrics(config)
        split_metrics = metrics_fn.compute_all_metrics(embeddings, labels)
        
        prefixed_metrics = {}
        for key, value in split_metrics.items():
            prefixed_metrics[f"{split_name}_{key}"] = value
        
        return prefixed_metrics
    
    val_seen_metrics = compute_split_metrics_from_embeddings(val_seen_embeddings, val_seen_labels, "val_seen", config)
    test_seen_metrics = compute_split_metrics_from_embeddings(test_seen_embeddings, test_seen_labels, "test_seen", config)
    test_unseen_metrics = compute_split_metrics_from_embeddings(test_unseen_embeddings, test_unseen_labels, "test_unseen", config)
    
    all_metrics = {}
    all_metrics.update(val_seen_metrics)
    all_metrics.update(test_seen_metrics)
    all_metrics.update(test_unseen_metrics)
    
    if test_seen_metrics and test_unseen_metrics:
        generalization_gap_grouped_recall_at_k = test_seen_metrics.get("test_seen_grouped_recall_at_k", 0) - test_unseen_metrics.get("test_unseen_grouped_recall_at_k", 0)
        all_metrics["generalization_gap_grouped_recall_at_k"] = generalization_gap_grouped_recall_at_k

    metrics = all_metrics

    metrics["run_id"] = run_spec.run_id
    metrics["dataset_id"] = run_spec.dataset.dataset_id
    metrics["mode"] = "real"
    metrics["simulated"] = False
    metrics["warning"] = None

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return metrics


def run_contrastive_experiment(run_spec: RunSpec, output_dir: Path) -> Dict[str, Any]:
    """Run a real contrastive learning experiment."""
    return run_real_experiment(run_spec, output_dir)
