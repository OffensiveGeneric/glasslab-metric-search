"""
HPO module: Hyperparameter optimization for DML with Optuna + Syne Tune
"""

import optuna
from optuna import Trial, TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import os
from config import Config
from src.data.dataset import get_dataloaders
from src.models.backbone import ModelFactory
from src.losses.losses import SupervisedContrastiveLoss, TripletLoss, ShadowLoss
from src.metrics.metrics import AdvancedMetrics


class DMLHPO:
    """Hyperparameter optimization for Deep Metric Learning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dataloaders = get_dataloaders(config)
        self.device = config.training.device
        self.n_trials = config.hpo.n_trials
        self.study_name = "dml_optimization"
        
    def objective(self, trial: Trial) -> float:
        """Optimization objective for trial"""
        # Sample hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        margin = trial.suggest_float("margin", 0.1, 1.0)
        temperature = trial.suggest_float("temperature", 0.01, 0.2)
        
        # Create dataloader with sampled batch size
        train_loader = self.dataloaders["train_seen_0"]
        train_loader.batch_size = batch_size
        
        # Create model
        model = ModelFactory.create_backbone(self.config, "resnet18")
        model = model.to(self.device)
        
        # Choose loss function
        loss_choice = trial.suggest_categorical("loss_function", ["contrastive", "triplet", "shadow"])
        
        if loss_choice == "contrastive":
            loss_fn = SupervisedContrastiveLoss(temperature=temperature)
        elif loss_choice == "triplet":
            loss_fn = TripletLoss(margin=margin)
        else:
            loss_fn = ShadowLoss(embedding_dim=self.config.model.embedding_dim)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Training loop
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            embeddings = model(images)
            
            # Compute loss
            if loss_choice in ["contrastive", "shadow"]:
                loss = loss_fn(embeddings, labels)
            else:
                # Triplet loss needs anchor, positive, negative
                # Simple implementation: use random triplets
                loss = self._compute_triplet_loss(embeddings, labels, margin)
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Early pruning
            if batch_idx > 10:
                break
                
        return epoch_loss / min(len(train_loader), 10)
    
    def _compute_triplet_loss(self, embeddings: torch.Tensor,
                              labels: torch.Tensor,
                              margin: float) -> torch.Tensor:
        """Compute triplet loss with random sampling"""
        batch_size = embeddings.shape[0]
        
        # Simple random triplet sampling
        anchor_idx = list(range(batch_size))
        positive_idx = [(i + 1) % batch_size for i in anchor_idx]
        
        # Find negative indices (different class)
        negative_idx = []
        for i in anchor_idx:
            anchor_class = labels[i].item()
            neg_candidates = [j for j in range(batch_size) if labels[j].item() != anchor_class]
            if len(neg_candidates) > 0:
                negative_idx.append(neg_candidates[0])
            else:
                negative_idx.append((i + 1) % batch_size)
                
        anchor = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]
        
        # Triplet loss
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + margin)
        
        return loss.mean()
    
    def run_optimization(self) -> optuna.Study:
        """Run the hyperparameter optimization"""
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return study
    
    def get_best_config(self, study: optuna.Study) -> Dict[str, Any]:
        """Get best hyperparameters from study"""
        best_trial = study.best_trial
        return best_trial.params


class SyneTuneHPO:
    """Syne Tune-based distributed HPO for large-scale experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        from syne_tune import StoppingCriterion
        from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
        
        # Define search space
        self.search_space = {
            "batch_size": "[64, 128, 256, 512, 1024]",
            "learning_rate": "loguniform(1e-5, 1e-3)",
            "margin": "uniform(0.1, 1.0)"
        }
        
        # Define stopping criterion
        self.stopping_criterion = StoppingCriterion(
            max_resource=self.config.training.epochs
        )
        
        # Define scheduler
        self.scheduler = HyperbandScheduler(
            max_t=self.config.training.epochs,
            reduction_factor=3,
            resource_attr="epoch"
        )
        
    def run(self):
        """Run distributed HPO with Syne Tune"""
        # Placeholder for distributed execution
        print("Syne Tune HPO initialized")
        print(f"Search space: {self.search_space}")
        print(f"Scheduler: {self.scheduler}")
