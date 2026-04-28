from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        mask = mask - torch.eye(batch_size).float().to(device)
        
        similarity_matrix = similarity_matrix / self.temperature
        log_sim_matrix = similarity_matrix - torch.logsumexp(
            similarity_matrix, dim=1, keepdim=True
        )
        
        loss = -torch.sum(log_sim_matrix * mask) / torch.sum(mask)
        
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3, miner: str = "semi_hard"):
        super().__init__()
        self.margin = margin
        self.miner = miner
        
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        if self.miner == "semi_hard":
            loss = F.relu(pos_dist - neg_dist + self.margin)
            loss = loss[loss > 0]
            
        elif self.miner == "hard":
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        else:
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        return loss.mean() if len(loss) > 0 else torch.tensor(0.0).to(anchor.device)


class ShadowLoss(nn.Module):
    def __init__(self, embedding_dim: int = 512, 
                 projection_dim: int = 1,
                 learnable_projection: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.learnable_projection = learnable_projection
        
        if learnable_projection:
            self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
        else:
            proj_matrix = torch.randn(embedding_dim, projection_dim)
            self.register_buffer("projection_matrix", proj_matrix)
            
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.learnable_projection:
            return self.projection(embeddings)
        else:
            return embeddings @ self.projection_matrix
            
    def forward(self, anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        anchor_proj = self.project(anchor)
        positive_proj = self.project(positive)
        negative_proj = self.project(negative)
        
        pos_dist = torch.abs(anchor_proj - positive_proj)
        neg_dist = torch.abs(anchor_proj - negative_proj)
        
        margin = 0.3
        loss = F.relu(pos_dist - neg_dist + margin)
        
        return loss.mean()


class HardNegativeMiner:
    def __init__(self, margin: float = 0.3):
        self.margin = margin
        
    def mine(self, anchor: torch.Tensor,
             positive: torch.Tensor,
             negative: torch.Tensor,
             labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = anchor.shape[0]
        
        distances = self._compute_distance_matrix(anchor, negative)
        
        hard_triplets = []
        
        for i in range(batch_size):
            anchor_class = labels[i].item()
            
            pos_dist = F.pairwise_distance(anchor[i].unsqueeze(0), 
                                           positive[i].unsqueeze(0))
            
            neg_indices = [j for j in range(batch_size) 
                           if labels[j].item() != anchor_class]
            
            if len(neg_indices) == 0:
                continue
                
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
            for i in range(batch_size):
                anchor_class = labels[i].item()
                neg_indices = [j for j in range(batch_size) 
                               if labels[j].item() != anchor_class]
                if len(neg_indices) > 0:
                    hardest_neg = neg_indices[0]
                    hard_triplets.append((i, i, hardest_neg))
                    
        hard_anchor = torch.stack([anchor[i] for i, _, _ in hard_triplets])
        hard_positive = torch.stack([positive[i] for i, _, _ in hard_triplets])
        hard_negative = torch.stack([negative[j] for _, _, j in hard_triplets])
        
        return hard_anchor, hard_positive, hard_negative
        
    def _compute_distance_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        ab = a @ b.t()
        
        distances = a_sq + b_sq.t() - 2 * ab
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-8)
        
        return distances
