"""
Baseline experiments for metric learning evaluation
Tests random embeddings, frozen backbones (ResNet, DINO, CLIP) to validate evaluation pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fix FAISS OpenMP deadlock on macOS
if os.uname().sysname == "Darwin":
    os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import Cifar100Splitter
from src.models.backbone import Backbone
from src.metrics.metrics import AdvancedMetrics
import timm


def get_random_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100) -> dict:
    """Generate random Gaussian embeddings for baseline"""
    print("Generating random embeddings...")
    random_embeddings = {}
    
    for split_name in ["val_seen_0", "test_seen_0", "test_unseen_0"]:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            random_embeddings[f"{split_name}_embeddings"] = None
            random_embeddings[f"{split_name}_labels"] = None
            continue
            
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_eval_batches:
                    break
                # Random embeddings: same shape as typical embedding (e.g., 512-dim)
                batch_size = images.shape[0]
                random_emb = torch.randn(batch_size, 512)
                all_embeddings.append(random_emb)
                all_labels.append(labels)
        
        if all_embeddings:
            random_embeddings[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            random_embeddings[f"{split_name}_labels"] = torch.cat(all_labels)
        else:
            random_embeddings[f"{split_name}_embeddings"] = None
            random_embeddings[f"{split_name}_labels"] = None
    
    return random_embeddings


def get_frozen_resnet_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100) -> dict:
    """Generate embeddings from frozen ResNet50"""
    print("Loading frozen ResNet50...")
    backbone = Backbone(
        backbone_name="resnet50",
        pretrained=True,
        embedding_dim=512,
        freeze=True  # Freeze backbone
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    embeddings_dict = {}
    
    for split_name in ["val_seen_0", "test_seen_0", "test_unseen_0"]:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
            continue
            
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_eval_batches:
                    break
                images = images.to(device)
                embeddings = backbone(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    return embeddings_dict


def get_frozen_dino_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100) -> dict:
    """Generate embeddings from frozen DINO ViT"""
    print("Loading frozen DINO ViT...")
    backbone = Backbone(
        backbone_name="vit_base_patch16.dino",
        pretrained=True,
        embedding_dim=512,
        freeze=True  # Freeze backbone
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    embeddings_dict = {}
    
    for split_name in ["val_seen_0", "test_seen_0", "test_unseen_0"]:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
            continue
            
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_eval_batches:
                    break
                images = images.to(device)
                embeddings = backbone(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    return embeddings_dict


def get_frozen_clip_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100) -> dict:
    """Generate embeddings from frozen CLIP"""
    print("Loading frozen CLIP...")
    backbone = Backbone(
        backbone_name="clip-vit-base-patch32",
        pretrained=True,
        embedding_dim=512,
        freeze=True  # Freeze backbone
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    embeddings_dict = {}
    
    for split_name in ["val_seen_0", "test_seen_0", "test_unseen_0"]:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
            continue
            
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_eval_batches:
                    break
                images = images.to(device)
                embeddings = backbone(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    return embeddings_dict


def compute_metrics_for_embeddings(embeddings_dict: dict, config: Config, split_name: str) -> dict:
    """Compute metrics for a single split's embeddings"""
    embeddings = embeddings_dict.get(f"{split_name}_embeddings")
    labels = embeddings_dict.get(f"{split_name}_labels")
    
    if embeddings is None or labels is None:
        return {}
    
    metrics_fn = AdvancedMetrics(config)
    split_metrics = metrics_fn.compute_all_metrics(embeddings, labels)
    
    prefixed_metrics = {}
    for key, value in split_metrics.items():
        prefixed_metrics[f"{split_name}_{key}"] = value
    
    return prefixed_metrics


def run_baseline_experiment(baseline_name: str, embeddings_func, output_dir: Path, 
                           max_eval_batches: int = 100) -> dict:
    """Run a baseline experiment and return metrics"""
    print(f"\n{'='*60}")
    print(f"Running baseline: {baseline_name}")
    print(f"{'='*60}")
    
    # Load config
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    config = Config.from_yaml(str(config_path))
    
    # Create dataloaders
    splitter = Cifar100Splitter(config)
    dataloaders = splitter.create_dataloaders()
    
    # Get embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embeddings_dict = embeddings_func(dataloaders, device, max_eval_batches)
    
    # Compute metrics
    all_metrics = {}
    
    val_seen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "val_seen_0")
    test_seen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "test_seen_0")
    test_unseen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "test_unseen_0")
    
    all_metrics.update(val_seen_metrics)
    all_metrics.update(test_seen_metrics)
    all_metrics.update(test_unseen_metrics)
    
    # Compute generalization gap
    if test_seen_metrics and test_unseen_metrics:
        generalization_gap = test_seen_metrics.get("test_seen_grouped_recall_at_k", 0) - \
                            test_unseen_metrics.get("test_unseen_grouped_recall_at_k", 0)
        all_metrics["generalization_gap_grouped_recall_at_k"] = generalization_gap
    
    # Add baseline identifier
    all_metrics["baseline"] = baseline_name
    all_metrics["mode"] = "baseline"
    all_metrics["simulated"] = False
    
    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{baseline_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    
    # Save report
    report_path = output_dir / f"{baseline_name}_report.md"
    report_content = (
        f"# Baseline Experiment Report: {baseline_name}\n\n"
        f"- baseline: `{baseline_name}`\n"
        f"- mode: {all_metrics.get('mode', 'baseline')}\n"
        f"- simulated: {all_metrics.get('simulated', False)}\n"
        f"\n## Per-Split Metrics\n\n"
        f"- val_seen_grouped_recall_at_k: `{val_seen_metrics.get('val_seen_grouped_recall_at_k', 'N/A')}`\n"
        f"- test_seen_grouped_recall_at_k: `{test_seen_metrics.get('test_seen_grouped_recall_at_k', 'N/A')}`\n"
        f"- test_unseen_grouped_recall_at_k: `{test_unseen_metrics.get('test_unseen_grouped_recall_at_k', 'N/A')}`\n"
        f"- generalization_gap_grouped_recall_at_k: `{all_metrics.get('generalization_gap_grouped_recall_at_k', 'N/A')}`\n"
        f"- test_unseen_composite_score: `{all_metrics.get('test_unseen_composite_score', 'N/A')}`\n"
    )
    report_path.write_text(report_content, encoding="utf-8")
    
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--baseline", type=str, required=True,
                       choices=["random", "resnet50", "dino", "clip"],
                       help="Which baseline to run")
    parser.add_argument("--output-dir", type=str, default="baselines",
                       help="Output directory for results")
    parser.add_argument("--max-eval-batches", type=int, default=100,
                       help="Maximum evaluation batches")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.baseline
    
    # Map baseline name to function
    baseline_funcs = {
        "random": lambda: run_baseline_experiment("random", get_random_embeddings, output_dir, args.max_eval_batches),
        "resnet50": lambda: run_baseline_experiment("frozen_resnet50", get_frozen_resnet_embeddings, output_dir, args.max_eval_batches),
        "dino": lambda: run_baseline_experiment("frozen_dino", get_frozen_dino_embeddings, output_dir, args.max_eval_batches),
        "clip": lambda: run_baseline_experiment("frozen_clip", get_frozen_clip_embeddings, output_dir, args.max_eval_batches),
    }
    
    metrics = baseline_funcs[args.baseline]()
    
    print(f"\n{'='*60}")
    print(f"Baseline {args.baseline} completed!")
    print(f"{'='*60}")
    print(f"test_unseen_grouped_recall_at_k: {metrics.get('test_unseen_grouped_recall_at_k', 'N/A')}")
    print(f"test_unseen_composite_score: {metrics.get('test_unseen_composite_score', 'N/A')}")
    print(f"generalization_gap_grouped_recall_at_k: {metrics.get('generalization_gap_grouped_recall_at_k', 'N/A')}")
    
    return metrics


if __name__ == "__main__":
    main()
