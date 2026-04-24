"""
L2A-NC module: Learning to Augment Novel Classes via synthetic generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class L2ANCGenerator(nn.Module):
    """Conditional generator for synthetic embedding augmentation"""
    
    def __init__(self, latent_dim: int = 128,
                 embedding_dim: int = 512,
                 num_classes: int = 100,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Generator architecture
        layers = []
        input_dim = latent_dim * 2  # latent + class embedding
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else embedding_dim
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.ReLU())
                
            input_dim = output_dim
            
        self.generator = nn.Sequential(*layers)
        
    def forward(self, latent: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Generate synthetic embeddings"""
        class_emb = self.class_embedding(class_labels)
        inputs = torch.cat([latent, class_emb], dim=1)
        synthetic_embeddings = self.generator(inputs)
        return synthetic_embeddings
    
    def sample(self, num_samples: int, class_labels: torch.Tensor,
              device: str = "cuda") -> torch.Tensor:
        """Sample synthetic embeddings"""
        latent = torch.randn(num_samples, self.latent_dim).to(device)
        class_labels = class_labels.to(device)
        return self.forward(latent, class_labels)


class L2ANCLoss(nn.Module):
    """Loss for L2A-NC: KL divergence between real and synthetic distributions"""
    
    def __init__(self, kl_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
        
    def forward(self, real_embeddings: torch.Tensor,
                synthetic_embeddings: torch.Tensor,
                class_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KL divergence loss between real and synthetic distributions
        """
        # Normalize embeddings
        real_normalized = F.normalize(real_embeddings, dim=1)
        synthetic_normalized = F.normalize(synthetic_embeddings, dim=1)
        
        # Compute pairwise similarity matrices
        real_sim = real_normalized @ real_normalized.t()
        synthetic_sim = synthetic_normalized @ synthetic_normalized.t()
        
        # KL divergence between similarity distributions
        # Use softmax for probability distribution
        real_prob = F.softmax(real_sim, dim=1)
        synthetic_prob = F.softmax(synthetic_sim, dim=1)
        
        # KL(P||Q) = sum(P * log(P/Q))
        kl_loss = F.kl_div(
            torch.log(synthetic_prob + 1e-8),
            real_prob,
            reduction="batchmean"
        )
        
        total_loss = self.kl_weight * kl_loss
        
        return total_loss, {"kl_divergence": kl_loss.item()}


class L2ANCModule:
    """Complete L2A-NC module for synthetic class augmentation"""
    
    def __init__(self, config):
        self.config = config
        self.generator = L2ANCGenerator(
            latent_dim=config.l2anc.latent_dim,
            embedding_dim=config.model.embedding_dim,
            num_classes=config.data.num_classes,
            hidden_dim=config.l2anc.generator_hidden_dim,
            num_layers=config.l2anc.generator_layers
        )
        self.loss_fn = L2ANCLoss(kl_weight=config.l2anc.kl_divergence_weight)
        self.optimizer = None
        
    def to(self, device: str):
        """Move module to device"""
        self.generator = self.generator.to(device)
        return self
        
    def initialize_optimizer(self, lr: float = 1e-4):
        """Initialize optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=lr, weight_decay=1e-4
        )
        
    def train_step(self, real_embeddings: torch.Tensor,
                   class_labels: torch.Tensor,
                   device: str = "cuda") -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step"""
        self.generator.train()
        self.optimizer.zero_grad()
        
        # Generate synthetic embeddings
        synthetic_embeddings = self.generator.sample(
            num_samples=len(real_embeddings),
            class_labels=class_labels,
            device=device
        )
        
        # Compute loss
        loss, metrics = self.loss_fn(real_embeddings, synthetic_embeddings, class_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), metrics
    
    def generate_synthetic_samples(self, num_samples_per_class: int,
                                   class_labels: torch.Tensor,
                                   device: str = "cuda") -> torch.Tensor:
        """Generate synthetic samples for specific classes"""
        self.generator.eval()
        with torch.no_grad():
            synthetic = self.generator.sample(
                num_samples=num_samples_per_class * len(class_labels),
                class_labels=class_labels.repeat_interleave(num_samples_per_class),
                device=device
            )
        return synthetic
