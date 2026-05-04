"""
Baseline experiments for metric learning evaluation
Tests random embeddings, frozen backbones (ResNet, DINO, CLIP) to validate evaluation pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path

# Log image commit if available
image_commit = os.environ.get("GLASSLAB_IMAGE_COMMIT", "").strip()
if image_commit:
    print(f"Image commit: {image_commit}")

# Fix FAISS OpenMP deadlock on macOS
if os.uname().sysname == "Darwin":
    os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import Cifar100Splitter
from src.models.backbone import Backbone
from src.metrics.metrics import AdvancedMetrics
import timm


def get_random_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100, db_path: Optional[Path] = None) -> dict:
    """Generate random Gaussian embeddings for baseline"""
    print("Generating random embeddings...")
    random_embeddings = {}
    conn = None
    cursor = None
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS random_embeddings (
                split_name TEXT,
                batch_idx INTEGER,
                embeddings BLOB,
                labels BLOB,
                PRIMARY KEY (split_name, batch_idx)
            )
        """)
    
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
                
                # Store in SQLite if db_path provided
                if cursor:
                    embeddings_np = random_emb.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    cursor.execute(
                        "INSERT OR REPLACE INTO random_embeddings (split_name, batch_idx, embeddings, labels) VALUES (?, ?, ?, ?)",
                        (split_name, batch_idx, embeddings_np.tobytes(), labels_np.tobytes())
                    )
                    conn.commit()
        
        if all_embeddings:
            random_embeddings[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            random_embeddings[f"{split_name}_labels"] = torch.cat(all_labels)
            # Free memory
            del all_embeddings
            del all_labels
            torch.cuda.empty_cache()
        else:
            random_embeddings[f"{split_name}_embeddings"] = None
            random_embeddings[f"{split_name}_labels"] = None
    
    if conn:
        conn.close()
    
    return random_embeddings


def get_frozen_resnet_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100, db_path: Optional[Path] = None) -> dict:
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
    conn = None
    cursor = None
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resnet50_embeddings (
                split_name TEXT,
                batch_idx INTEGER,
                embeddings BLOB,
                labels BLOB,
                PRIMARY KEY (split_name, batch_idx)
            )
        """)
    
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
                
                # Store in SQLite if db_path provided
                if cursor:
                    emb_cpu = embeddings.cpu()
                    embeddings_np = emb_cpu.numpy()
                    labels_np = labels.cpu().numpy()
                    cursor.execute(
                        "INSERT OR REPLACE INTO resnet50_embeddings (split_name, batch_idx, embeddings, labels) VALUES (?, ?, ?, ?)",
                        (split_name, batch_idx, embeddings_np.tobytes(), labels_np.tobytes())
                    )
                    conn.commit()
                
                # Free GPU memory after each batch
                del embeddings
                torch.cuda.empty_cache()
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
            # Free memory
            del all_embeddings
            del all_labels
            torch.cuda.empty_cache()
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    if conn:
        conn.close()
    
    return embeddings_dict


def get_frozen_dino_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100, db_path: Optional[Path] = None) -> dict:
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
    conn = None
    cursor = None
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dino_embeddings (
                split_name TEXT,
                batch_idx INTEGER,
                embeddings BLOB,
                labels BLOB,
                PRIMARY KEY (split_name, batch_idx)
            )
        """)
    
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
                
                # Store in SQLite if db_path provided
                if cursor:
                    emb_cpu = embeddings.cpu()
                    embeddings_np = emb_cpu.numpy()
                    labels_np = labels.cpu().numpy()
                    cursor.execute(
                        "INSERT OR REPLACE INTO dino_embeddings (split_name, batch_idx, embeddings, labels) VALUES (?, ?, ?, ?)",
                        (split_name, batch_idx, embeddings_np.tobytes(), labels_np.tobytes())
                    )
                    conn.commit()
                
                # Free GPU memory after each batch
                del embeddings
                torch.cuda.empty_cache()
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
            # Free memory
            del all_embeddings
            del all_labels
            torch.cuda.empty_cache()
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    if conn:
        conn.close()
    
    return embeddings_dict


def get_frozen_clip_embeddings(dataloaders: dict, device: str, max_eval_batches: int = 100, db_path: Optional[Path] = None) -> dict:
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
    conn = None
    cursor = None
    if db_path:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clip_embeddings (
                split_name TEXT,
                batch_idx INTEGER,
                embeddings BLOB,
                labels BLOB,
                PRIMARY KEY (split_name, batch_idx)
            )
        """)
    
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
                
                # Store in SQLite if db_path provided
                if cursor:
                    emb_cpu = embeddings.cpu()
                    embeddings_np = emb_cpu.numpy()
                    labels_np = labels.cpu().numpy()
                    cursor.execute(
                        "INSERT OR REPLACE INTO clip_embeddings (split_name, batch_idx, embeddings, labels) VALUES (?, ?, ?, ?)",
                        (split_name, batch_idx, embeddings_np.tobytes(), labels_np.tobytes())
                    )
                    conn.commit()
                
                # Free GPU memory after each batch
                del embeddings
                torch.cuda.empty_cache()
        
        if all_embeddings:
            embeddings_dict[f"{split_name}_embeddings"] = torch.cat(all_embeddings)
            embeddings_dict[f"{split_name}_labels"] = torch.cat(all_labels)
            # Free memory
            del all_embeddings
            del all_labels
            torch.cuda.empty_cache()
        else:
            embeddings_dict[f"{split_name}_embeddings"] = None
            embeddings_dict[f"{split_name}_labels"] = None
    
    if conn:
        conn.close()
    
    return embeddings_dict




def compute_metrics_for_embeddings(embeddings_dict: dict, config: Config, split_name: str) -> dict:
    """Compute metrics for a single split's embeddings"""
    embeddings = embeddings_dict.get(f"{split_name}_embeddings")
    labels = embeddings_dict.get(f"{split_name}_labels")
    
    if embeddings is None or labels is None:
        return {}
    
    metrics_fn = AdvancedMetrics(config)
    split_metrics = metrics_fn.compute_all_metrics(embeddings, labels)
    
    # Compute random embedding baseline
    generator = torch.Generator().manual_seed(int(config.data.seed) + 17)
    random_embeddings = torch.randn(
        embeddings.shape,
        generator=generator,
        dtype=embeddings.dtype,
    )
    random_metrics = metrics_fn.compute_all_metrics(random_embeddings, labels)
    
    prefixed_metrics = {}
    for key, value in split_metrics.items():
        prefixed_metrics[f"{split_name}_{key}"] = value
    
    for key, value in random_metrics.items():
        prefixed_metrics[f"{split_name}_random_embedding_{key}"] = value
    
    # Compute lift metrics for random embeddings
    real_recall = prefixed_metrics.get(f"{split_name}_grouped_recall_at_k")
    random_recall = prefixed_metrics.get(f"{split_name}_random_embedding_grouped_recall_at_k")
    
    if real_recall is not None and random_recall is not None:
        prefixed_metrics[f"{split_name}_grouped_recall_lift_vs_random_embeddings"] = real_recall - random_recall
    
    return prefixed_metrics




def run_baseline_experiment(baseline_name: str, embeddings_func, output_dir: Path, 
                            max_eval_batches: int = 100, db_type: Literal["sqlite", "postgresql"] = "sqlite",
                            db_path: Optional[str] = None, run_shuffled: bool = False) -> dict:
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
    print(f"Using database: {db_type}")
    if db_path:
        print(f"Database path: {db_path}")
    
    embeddings_dict = embeddings_func(dataloaders, device, max_eval_batches, db_path=Path(db_path) if db_path else None)
    
    # Compute metrics
    all_metrics = {}
    
    val_seen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "val_seen_0")
    test_seen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "test_seen_0")
    test_unseen_metrics = compute_metrics_for_embeddings(embeddings_dict, config, "test_unseen_0")
    
    all_metrics.update(val_seen_metrics)
    all_metrics.update(test_seen_metrics)
    all_metrics.update(test_unseen_metrics)
    
    # Add shuffled label baseline if requested
    if run_shuffled:
        print(f"\n{'='*60}")
        print(f"Running shuffled label baseline for comparison")
        print(f"{'='*60}")
        
        # Shuffle labels for each split
        def shuffle_labels(labels, seed_offset=0):
            generator = torch.Generator().manual_seed(int(config.data.seed) + seed_offset)
            shuffled = labels[torch.randperm(labels.numel(), generator=generator)]
            return shuffled
        
        shuffled_val_labels = shuffle_labels(embeddings_dict["val_seen_0_labels"] if embeddings_dict.get("val_seen_0_labels") is not None else torch.empty(0), seed_offset=1000)
        shuffled_test_seen_labels = shuffle_labels(embeddings_dict["test_seen_0_labels"] if embeddings_dict.get("test_seen_0_labels") is not None else torch.empty(0), seed_offset=1001)
        shuffled_test_unseen_labels = shuffle_labels(embeddings_dict["test_unseen_0_labels"] if embeddings_dict.get("test_unseen_0_labels") is not None else torch.empty(0), seed_offset=1002)
        
        shuffled_val_embeddings = embeddings_dict.get("val_seen_0_embeddings")
        shuffled_test_seen_embeddings = embeddings_dict.get("test_seen_0_embeddings")
        shuffled_test_unseen_embeddings = embeddings_dict.get("test_unseen_0_embeddings")
        
        if shuffled_val_embeddings is not None:
            shuffled_val_metrics = compute_metrics_for_embeddings({"val_seen_0_embeddings": shuffled_val_embeddings, "val_seen_0_labels": shuffled_val_labels}, config, "val_seen_0")
            all_metrics.update(shuffled_val_metrics)
        
        if shuffled_test_seen_embeddings is not None:
            shuffled_test_seen_metrics = compute_metrics_for_embeddings({"test_seen_0_embeddings": shuffled_test_seen_embeddings, "test_seen_0_labels": shuffled_test_seen_labels}, config, "test_seen_0")
            all_metrics.update(shuffled_test_seen_metrics)
        
        if shuffled_test_unseen_embeddings is not None:
            shuffled_test_unseen_metrics = compute_metrics_for_embeddings({"test_unseen_0_embeddings": shuffled_test_unseen_embeddings, "test_unseen_0_labels": shuffled_test_unseen_labels}, config, "test_unseen_0")
            all_metrics.update(shuffled_test_unseen_metrics)
            
            # Compute lift over shuffled labels
            if test_unseen_metrics and shuffled_test_unseen_metrics:
                shuffled_recall = shuffled_test_unseen_metrics.get("test_unseen_grouped_recall_at_k", 0)
                real_recall = test_unseen_metrics.get("test_unseen_grouped_recall_at_k", 0)
                lift_vs_shuffled = real_recall - shuffled_recall
                all_metrics["test_unseen_grouped_recall_lift_vs_shuffled_labels"] = lift_vs_shuffled
                print(f"Lift over shuffled labels: {lift_vs_shuffled:.4f}")
    
    # Add baseline identifier
    all_metrics["baseline"] = baseline_name
    all_metrics["mode"] = "baseline"
    all_metrics["simulated"] = False
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    all_metrics = convert_to_native(all_metrics)
    
    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{baseline_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    
    # Save report
    report_path = output_dir / f"{baseline_name}_report.md"
    
    # Compute shuffled metrics for report if available
    shuffled_test_unseen_recall = all_metrics.get("test_unseen_shuffled_grouped_recall_at_k", "N/A")
    lift_vs_shuffled = all_metrics.get("test_unseen_grouped_recall_lift_vs_shuffled_labels", "N/A")
    
    random_test_unseen_recall = all_metrics.get("test_unseen_random_embedding_grouped_recall_at_k", "N/A")
    lift_vs_random = all_metrics.get("test_unseen_grouped_recall_lift_vs_random_embeddings", "N/A")
    
    report_content = (
        f"# Baseline Experiment Report: {baseline_name}\n\n"
        f"- baseline: `{baseline_name}`\n"
        f"- mode: {all_metrics.get('mode', 'baseline')}\n"
        f"- simulated: {all_metrics.get('simulated', False)}\n"
        f"\n## Per-Split Metrics\n\n"
        f"- val_seen_grouped_recall_at_k: `{val_seen_metrics.get('val_seen_grouped_recall_at_k', 'N/A')}`\n"
        f"- test_seen_grouped_recall_at_k: `{test_seen_metrics.get('test_seen_grouped_recall_at_k', 'N/A')}`\n"
        f"- test_unseen_grouped_recall_at_k: `{test_unseen_metrics.get('test_unseen_grouped_recall_at_k', 'N/A')}`\n"
        f"\n## Shuffled Label Baseline Comparison\n\n"
        f"- test_unseen_shuffled_grouped_recall_at_k: `{shuffled_test_unseen_recall}`\n"
        f"- test_unseen_grouped_recall_lift_vs_shuffled_labels: `{lift_vs_shuffled}`\n"
        f"\n## Random Embedding Baseline Comparison\n\n"
        f"- test_unseen_random_embedding_grouped_recall_at_k: `{random_test_unseen_recall}`\n"
        f"- test_unseen_grouped_recall_lift_vs_random_embeddings: `{lift_vs_random}`\n"
        f"\n## Gallery Information\n\n"
        f"- test_unseen_grouped_recall_at_k_gallery_size: `{all_metrics.get('test_unseen_grouped_recall_at_k_gallery_size', 'N/A')}`\n"
        f"- test_unseen_grouped_recall_at_k_gallery_classes: `{all_metrics.get('test_unseen_grouped_recall_at_k_gallery_classes', 'N/A')}`\n"
        f"- test_unseen_grouped_recall_at_k_partial: `{all_metrics.get('test_unseen_grouped_recall_at_k_partial', 'N/A')}`\n"
        f"\n## Generalization\n\n"
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
    parser.add_argument("--run-shuffled", action="store_true",
                       help="Also run shuffled label baseline for comparison")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.baseline
    
    # Map baseline name to function
    baseline_funcs = {
        "random": lambda: run_baseline_experiment("random", get_random_embeddings, output_dir, args.max_eval_batches, run_shuffled=args.run_shuffled),
        "resnet50": lambda: run_baseline_experiment("frozen_resnet50", get_frozen_resnet_embeddings, output_dir, args.max_eval_batches, run_shuffled=args.run_shuffled),
        "dino": lambda: run_baseline_experiment("frozen_dino", get_frozen_dino_embeddings, output_dir, args.max_eval_batches, run_shuffled=args.run_shuffled),
        "clip": lambda: run_baseline_experiment("frozen_clip", get_frozen_clip_embeddings, output_dir, args.max_eval_batches, run_shuffled=args.run_shuffled),
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
