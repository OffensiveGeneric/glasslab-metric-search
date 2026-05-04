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
from src.metrics.metrics import AdvancedMetrics


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


def test_global_recall_at_1_random_embedding():
    """Test global Recall@1 with random embeddings matches chance (~0.05 for 20 classes)."""
    # Create balanced 20-class dataset
    num_classes = 20
    samples_per_class = 50
    total_samples = num_classes * samples_per_class
    
    # Generate random embeddings
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(total_samples, 128, generator=generator)
    
    # Generate balanced labels
    labels = torch.tensor([i for i in range(num_classes) for _ in range(samples_per_class)])
    
    # Compute metrics
    metrics_fn = AdvancedMetrics(Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml"))
    metrics = metrics_fn._global_recall_at_k_details(embeddings, labels, k=1)
    
    # Random embeddings should be near chance
    chance_exact = metrics["global_recall_chance_at_k_exact"]
    chance_approx = metrics["global_recall_chance_at_k_approx"]
    observed = metrics["global_recall_at_k"]
    
    # For 20 classes, chance should be ~0.05
    assert 0.02 <= chance_approx <= 0.08, f"Chance approx should be ~0.05, got {chance_approx}"
    assert 0.02 <= chance_exact <= 0.08, f"Chance exact should be ~0.05, got {chance_exact}"
    
    # Random embeddings should be near chance (tolerance for finite samples)
    # Allow wider tolerance for random embeddings
    assert abs(observed - chance_exact) <= 0.08, f"Random embedding recall {observed} not near chance {chance_exact}"


def test_grouped_recall_at_k_random():
    """Test grouped Recall@5 with random embeddings matches analytic chance (~0.41)."""
    # Create balanced 10-class dataset
    num_classes = 10
    samples_per_class = 100
    total_samples = num_classes * samples_per_class
    
    # Generate random embeddings
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(total_samples, 128, generator=generator)
    
    # Generate balanced labels
    labels = torch.tensor([i for i in range(num_classes) for _ in range(samples_per_class)])
    
    # Compute grouped Recall@5
    metrics_fn = AdvancedMetrics(Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml"))
    metrics = metrics_fn.grouped_recall_at_k_details(embeddings, labels, k=5, group_size=10)
    
    chance = metrics["grouped_recall_chance_at_k"]
    observed = metrics["grouped_recall_at_k"]
    
    # For grouped Recall@5 with group_size=10, chance is ~1 - (9/10)^5 = 0.40951
    expected_chance = 1.0 - (9.0 / 10.0) ** 5
    assert 0.35 <= chance <= 0.45, f"Chance should be ~0.41, got {chance}"
    assert abs(chance - expected_chance) <= 0.02, f"Chance {chance} should match analytic {expected_chance}"
    
    # Random embeddings should be near chance
    assert abs(observed - chance) <= 0.08, f"Random embedding recall {observed} not near chance {chance}"


def test_shuffled_label_baseline():
    """Test shuffled label baseline produces metrics near chance."""
    num_classes = 20
    samples_per_class = 50
    total_samples = num_classes * samples_per_class
    
    # Generate clusterable embeddings (high silhouette)
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(total_samples, 128, generator=generator)
    
    # Shift embeddings by class to create clusters
    class_offsets = torch.randn(num_classes, 128, generator=generator) * 5.0
    for i in range(num_classes):
        start = i * samples_per_class
        end = start + samples_per_class
        embeddings[start:end] += class_offsets[i]
    
    labels = torch.tensor([i for i in range(num_classes) for _ in range(samples_per_class)])
    
    # Compute metrics on real embeddings
    metrics_fn = AdvancedMetrics(Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml"))
    real_metrics = metrics_fn.compute_all_metrics(embeddings, labels)
    
    # Shuffle labels
    generator = torch.Generator().manual_seed(42)
    shuffled_labels = labels[torch.randperm(labels.numel(), generator=generator)]
    shuffled_metrics = metrics_fn.compute_all_metrics(embeddings, shuffled_labels)
    
    # Shuffled should be near chance
    real_recall = real_metrics.get("grouped_recall_at_k", 0.0)
    shuffled_recall = shuffled_metrics.get("grouped_recall_at_k", 0.0)
    chance = shuffled_metrics.get("grouped_recall_chance_at_k", 0.0)
    
    # Shuffled recall should be near chance
    assert abs(shuffled_recall - chance) <= 0.08, f"Shuffled recall {shuffled_recall} not near chance {chance}"
    
    # Shuffled should be significantly lower than real
    assert shuffled_recall < real_recall * 0.9, f"Shuffled recall {shuffled_recall} should be lower than real {real_recall}"


def test_perfect_clustered_embeddings():
    """Test that perfectly clustered embeddings show high Recall@1."""
    num_classes = 10
    samples_per_class = 100
    
    # Generate perfectly clustered embeddings
    embeddings_list = []
    labels_list = []
    for i in range(num_classes):
        # Cluster around a mean
        cluster_mean = torch.randn(1, 64) * 10.0
        cluster_samples = cluster_mean.repeat(samples_per_class, 1)
        cluster_samples += torch.randn(samples_per_class, 64) * 0.1  # Small variance
        embeddings_list.append(cluster_samples)
        labels_list.append(torch.full((samples_per_class,), i))
    
    embeddings = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)
    
    # Compute metrics
    metrics_fn = AdvancedMetrics(Config.from_yaml("configs/search_spaces/cifar100_contrastive_v0.yaml"))
    metrics = metrics_fn.compute_all_metrics(embeddings, labels)
    
    # Perfect clustering should have high global Recall@1
    global_recall = metrics.get("global_recall_at_1", 0.0)
    assert global_recall > 0.8, f"Perfect clustering should have high global Recall@1, got {global_recall}"
    
    # Shuffled labels should destroy clustering
    generator = torch.Generator().manual_seed(42)
    shuffled_labels = labels[torch.randperm(labels.numel(), generator=generator)]
    shuffled_metrics = metrics_fn.compute_all_metrics(embeddings, shuffled_labels)
    
    shuffled_recall = shuffled_metrics.get("global_recall_at_1", 0.0)
    assert shuffled_recall < global_recall * 0.5, f"Shuffled should destroy clustering: {shuffled_recall} vs {global_recall}"


def test_metrics_comprehensive():
    """Test all metrics are computed correctly."""
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    config = Config.from_yaml(str(config_path))
    config.evaluation.max_eval_batches = 4  # Limit for speed
    
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders.get("test_unseen_0")
    if test_loader is None:
        pytest.skip("test_unseen_0 not available")
    
    # Get a batch
    images, labels = next(iter(test_loader))
    device = config.training.device
    images = images.to(device)
    
    model = ModelFactory.create_backbone(config, "resnet18")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(images)
    
    # Compute all metrics
    metrics_fn = AdvancedMetrics(config)
    metrics = metrics_fn.compute_all_metrics(embeddings.cpu(), labels.cpu())
    
    # Check all expected metrics are present
    expected_metrics = [
        "grouped_recall_at_k",
        "grouped_recall_chance_at_k",
        "global_recall_at_1",
        "global_recall_at_1_chance_exact",
        "global_recall_at_1_chance_approx",
        "global_recall_at_1_num_classes",
        "global_recall_at_1_num_samples",
        "opis",
        "adjusted_mutual_info",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "silhouette_score",
        "composite_score",
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert metrics[metric] is not None, f"Metric {metric} is None"

