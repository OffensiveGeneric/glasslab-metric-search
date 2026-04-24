"""
Model module: Backbones for contrastive learning (ResNet, ViT, CLIP, DINO)
"""

import torch
import torch.nn as nn
import timm
from transformers import CLIPVisionModel, DinoVisionModel
from typing import List, Dict


class Backbone(nn.Module):
    """Generic backbone wrapper for feature extraction"""
    
    def __init__(self, backbone_name: str, pretrained: bool = True, 
                 embedding_dim: int = 512, freeze: bool = False):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        
        if "resnet" in backbone_name.lower():
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0
            )
            if hasattr(self.backbone, "fc"):
                in_features = self.backbone.fc.in_features
            elif hasattr(self.backbone, "classifier"):
                in_features = self.backbone.classifier.in_features
            else:
                in_features = self.backbone.num_features
                
        elif "vit" in backbone_name.lower() or "clip" in backbone_name.lower():
            if "clip" in backbone_name.lower():
                self.backbone = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-base-patch32" if "base" in backbone_name else "openai/clip-vit-large-patch14"
                )
            elif "dino" in backbone_name.lower():
                self.backbone = DinoVisionModel.from_pretrained(
                    "facebook/dino-vitb16" if "base" in backbone_name else "facebook/dino-vitl16"
                )
            else:
                self.backbone = timm.create_model(
                    backbone_name, pretrained=pretrained, num_classes=0
                )
            
            if hasattr(self.backbone, "num_features"):
                in_features = self.backbone.num_features
            else:
                in_features = 768
                
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        if freeze:
            self.freeze_backbone()
            
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """Extract features and project to embedding space"""
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            embeddings = self.projector(features)
            return embeddings
        
    def get_backbone_features(self, x):
        """Get backbone features without projection"""
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            return features


class ModelFactory:
    """Factory for creating different backbones"""
    
    @staticmethod
    def create_backbone(config, backbone_name: str) -> Backbone:
        """Create a backbone model"""
        return Backbone(
            backbone_name=backbone_name,
            pretrained=config.model.pretrained,
            embedding_dim=config.model.embedding_dim,
            freeze=config.model.freeze_backbone
        )
    
    @staticmethod
    def create_all_backbones(config) -> Dict[str, Backbone]:
        """Create all backbones specified in config"""
        backbones = {}
        for backbone_name in config.model.backbones:
            backbones[backbone_name] = ModelFactory.create_backbone(config, backbone_name)
        return backbones
