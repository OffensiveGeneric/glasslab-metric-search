"""
Main training script for DML experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import yaml
from typing import Dict, Any

from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.backbone import ModelFactory
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss, HardNegativeMiner
from src.metrics.metrics import AdvancedMetrics, StatisticalTests


class DMLTrainer:
    """Main trainer for Deep Metric Learning experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.dataloaders = get_dataloaders(config)
        self.model = ModelFactory.create_backbone(config, "resnet18")
        self.model = self.model.to(self.device)
        
        # Loss functions
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=config.loss.contrastive["temperature"]
        )
        self.triplet_loss = TripletLoss(
            margin=config.loss.triplet["margin"],
            miner=config.loss.triplet["miner"]
        )
        self.shadow_loss = ShadowLoss(
            embedding_dim=config.model.embedding_dim,
            projection_dim=config.loss.shadow["projection_dim"],
            learnable_projection=config.loss.shadow["learnable_projection"]
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Metrics
        self.metrics = AdvancedMetrics(config)
        
        # Training state
        self.epoch = 0
        self.best_val_score = 0.0
        self.checkpoint_path = f"{config.project_dir}/results/checkpoints"
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
    def train_epoch(self, dataloader: DataLoader, loss_fn_type: str = "contrastive") -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            
            # Compute loss
            if loss_fn_type == "contrastive":
                loss = self.contrastive_loss(embeddings, labels)
            elif loss_fn_type == "triplet":
                # Use hard negative mining
                miner = HardNegativeMiner(margin=self.config.loss.triplet["margin"])
                anchor, positive, negative = miner.mine(embeddings, embeddings, embeddings, labels)
                loss = self.triplet_loss(anchor, positive, negative)
            elif loss_fn_type == "shadow":
                # For shadow loss, we need anchor, positive, negative
                # Simple implementation: use sequential samples
                batch_size = embeddings.shape[0]
                anchor = embeddings
                positive = torch.cat([embeddings[1:], embeddings[:1]])
                negative = torch.cat([embeddings[-1:], embeddings[:-1]])
                loss = self.shadow_loss(anchor, positive, negative)
            else:
                raise ValueError(f"Unknown loss function: {loss_fn_type}")
                
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limit batches for faster iteration
            if batch_idx > 100:
                break
                
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set"""
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        results = self.metrics.compute_all_metrics(all_embeddings, all_labels)
        
        return results
    
    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "score": score
        }
        
        filepath = f"{self.checkpoint_path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filepath)
        
        if is_best:
            torch.save(checkpoint, f"{self.checkpoint_path}/best_model.pt")
            
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        return checkpoint["score"]
    
    def run_experiment(self, loss_fn_type: str = "contrastive", epochs: int = 10) -> Dict[str, float]:
        """Run complete training and evaluation experiment"""
        train_loader = self.dataloaders["train_seen_0"]
        val_loader = self.dataloaders["val_seen_0"]
        test_seen_loader = self.dataloaders["test_seen_0"]
        test_unseen_loader = self.dataloaders["test_unseen_0"]
        
        results = {
            "train_loss": [],
            "val_metrics": [],
            "test_seen_metrics": [],
            "test_unseen_metrics": []
        }
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, loss_fn_type)
            results["train_loss"].append(train_loss)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Evaluate on validation
            val_metrics = self.evaluate(val_loader)
            results["val_metrics"].append(val_metrics)
            
            print(f"Val Metrics: {val_metrics}")
            
            # Evaluate on test sets
            test_seen_metrics = self.evaluate(test_seen_loader)
            test_unseen_metrics = self.evaluate(test_unseen_loader)
            
            results["test_seen_metrics"].append(test_seen_metrics)
            results["test_unseen_metrics"].append(test_unseen_metrics)
            
            # Save checkpoint
            is_best = val_metrics["grouped_recall_at_k"] > self.best_val_score
            self.best_val_score = max(self.best_val_score, val_metrics["grouped_recall_at_k"])
            self.save_checkpoint(epoch, self.best_val_score, is_best)
            
            # Early stopping check
            if epoch > 5:
                recent_scores = [m["grouped_recall_at_k"] for m in results["val_metrics"][-5:]]
                if max(recent_scores) <= self.best_val_score:
                    print("Early stopping triggered")
                    break
                    
        return results
    
    def run_baseline_evaluation(self) -> Dict[str, Dict[str, float]]:
        """Evaluate zero-shot baselines (CLIP, DINO, ResNet)"""
        results = {}
        
        for backbone_name in self.config.model.backbones:
            print(f"Evaluating {backbone_name}...")
            
            # Create model
            backbone = ModelFactory.create_backbone(self.config, backbone_name)
            backbone = backbone.to(self.device)
            backbone.eval()
            
            # Evaluate on all sets
            test_seen_loader = self.dataloaders["test_seen_0"]
            test_unseen_loader = self.dataloaders["test_unseen_0"]
            
            test_seen_metrics = self._evaluate_backbone(backbone, test_seen_loader)
            test_unseen_metrics = self._evaluate_backbone(backbone, test_unseen_loader)
            
            results[backbone_name] = {
                "test_seen": test_seen_metrics,
                "test_unseen": test_unseen_metrics
            }
            
            print(f"{backbone_name} results: {results[backbone_name]}")
            
        return results
    
    def _evaluate_backbone(self, backbone: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate a single backbone"""
        backbone.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                embeddings = backbone(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        return self.metrics.compute_all_metrics(all_embeddings, all_labels)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "triplet", "shadow"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation only")
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config) if args.config else Config()
    
    # Create trainer
    trainer = DMLTrainer(config)
    
    if args.baseline:
        # Run baseline evaluation
        results = trainer.run_baseline_evaluation()
        
        # Save results
        with open(f"{config.project_dir}/results/baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("Baseline evaluation complete!")
        print(f"Results saved to {config.project_dir}/results/baseline_results.json")
    else:
        # Run training experiment
        results = trainer.run_experiment(loss_fn_type=args.loss, epochs=args.epochs)
        
        # Save results
        with open(f"{config.project_dir}/results/training_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("Training complete!")
        print(f"Results saved to {config.project_dir}/results/training_results.json")


if __name__ == "__main__":
    main()
