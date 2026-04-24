"""
Losses module: Contrastive Loss, Triplet Loss, Shadow Loss, and mining strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import RetrievalMAP, RetrievalHitRate
from typing import Tuple, List, Optional
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Paper: https://arxiv.org/abs/2004.11362)"""
    
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Embedding features of shape (batch_size, embedding_dim)
            labels: Ground truth labels of shape (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Remove self-similarity
        mask = mask - torch.eye(batch_size).float().to(device)
        
        # Compute log probabilities
        similarity_matrix = similarity_matrix / self.temperature
        log_sim_matrix = similarity_matrix - torch.logsumexp(
            similarity_matrix, dim=1, keepdim=True
        )
        
        # Compute loss
        loss = -torch.sum(log_sim_matrix * mask) / torch.sum(mask)
        
        return loss


class TripletLoss(nn.Module):
    """Triplet Loss with mining options"""
    
    def __init__(self, margin: float = 0.3, miner: str = "semi_hard"):
        super().__init__()
        self.margin = margin
        self.miner = miner
        
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        if self.miner == "semi_hard":
            # Semi-hard negative mining: find negatives where positive < negative < margin
            loss = F.relu(pos_dist - neg_dist + self.margin)
            # Only keep losses > 0
            loss = loss[loss > 0]
            
        elif self.miner == "hard":
            # Hard negative mining: all negatives closer than margin
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        else:
            # Basic triplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        return loss.mean() if len(loss) > 0 else torch.tensor(0.0).to(anchor.device)


class ShadowLoss(nn.Module):
    """Memory-Linear Shadow Loss (1D projection for large batch training)"""
    
    def __init__(self, embedding_dim: int = 512, 
                 projection_dim: int = 1,
                 learnable_projection: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.learnable_projection = learnable_projection
        
        # Learnable projection matrix
        if learnable_projection:
            self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
        else:
            # Random projection
            proj_matrix = torch.randn(embedding_dim, projection_dim)
            self.register_buffer("projection_matrix", proj_matrix)
            
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to lower-dimensional space"""
        if self.learnable_projection:
            return self.projection(embeddings)
        else:
            return embeddings @ self.projection_matrix
            
    def forward(self, anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Shadow Loss: Compute loss in 1D projected space
        """
        # Project to 1D
        anchor_proj = self.project(anchor)
        positive_proj = self.project(positive)
        negative_proj = self.project(negative)
        
        # Compute distances in projected space
        pos_dist = torch.abs(anchor_proj - positive_proj)
        neg_dist = torch.abs(anchor_proj - negative_proj)
        
        # Triplet loss in 1D
        margin = 0.3
        loss = F.relu(pos_dist - neg_dist + margin)
        
        return loss.mean()


class HardNegativeMiner:
    """Hard negative mining for triplet loss"""
    
    def __init__(self, margin: float = 0.3):
        self.margin = margin
        
    def mine(self, anchor: torch.Tensor,
             positive: torch.Tensor,
             negative: torch.Tensor,
             labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard negatives for triplet loss
        """
        batch_size = anchor.shape[0]
        
        # Compute all pairwise distances
        distances = self._compute_distance_matrix(anchor, negative)
        
        hard_triplets = []
        
        for i in range(batch_size):
            anchor_class = labels[i].item()
            
            # Find hard negatives (closer to anchor than positive)
            pos_dist = F.pairwise_distance(anchor[i].unsqueeze(0), 
                                           positive[i].unsqueeze(0))
            
            # Get negatives from different classes
            neg_indices = [j for j in range(batch_size) 
                          if labels[j].item() != anchor_class]
            
            if len(neg_indices) == 0:
                continue
                
            # Find hardest negative
            hardest_neg = None
            max_dist_diff = -1
            
            for neg_idx in neg_indices:
                neg_dist = distances[i, neg_idx]
                dist_diff = neg_dist - pos_dist
                
                if dist_diff < self.margin and dist_diff > max_dist_diff:
                    hardest_neg = neg_idx
                    max_dist_diff = dist_diff
                    
            if hardest_neg is not None:
                hard_triplets.append((i, i, hardest_neg))
                
        if len(hard_triplets) == 0:
            # Fall back to semi-hard
            for i in range(batch_size):
                anchor_class = labels[i].item()
                neg_indices = [j for j in range(batch_size) 
                              if labels[j].item() != anchor_class]
                if len(neg_indices) > 0:
                    hardest_neg = neg_indices[0]
                    hard_triplets.append((i, i, hardest_neg))
                    
        # Create tensors
        hard_anchor = torch.stack([anchor[i] for i, _, _ in hard_triplets])
        hard_positive = torch.stack([positive[i] for i, _, _ in hard_triplets])
        hard_negative = torch.stack([negative[j] for _, _, j in hard_triplets])
        
        return hard_anchor, hard_positive, hard_negative
        
    def _compute_distance_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix"""
        # a: (N, D), b: (M, D) -> (N, M)
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        ab = a @ b.t()
        
        distances = a_sq + b_sq.t() - 2 * ab
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-8)
        
        return distances
