"""
Model module: Backbones for contrastive learning (ResNet, ViT, CLIP, DINO)
"""

import torch
import torch.nn as nn
import timm
from transformers import CLIPVisionModel
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
            # When num_classes=0, timm replaces fc with Identity
            # Get in_features from num_features attribute
            in_features = self.backbone.num_features
                
        elif "vit" in backbone_name.lower() or "clip" in backbone_name.lower():
            if "clip" in backbone_name.lower():
                self.backbone = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-base-patch32" if "base" in backbone_name else "openai/clip-vit-large-patch14"
                )
            elif "dino" in backbone_name.lower():
                # DINO is loaded via timm, not transformers
                self.backbone = timm.create_model(
                    "vit_base_patch16_224.dino" if "base" in backbone_name else "vit_large_patch16_224.dino",
                    pretrained=True, num_classes=0
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
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            embeddings = self.projector(features)
            return embeddings
        
    def get_backbone_features(self, x):
        """Get backbone features without projection"""
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            return features


class MLPBackbone(nn.Module):
    """Simple MLP backbone for synthetic smoke tests
    
    Takes feature tensors directly (no images) and projects to embedding space.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, embedding_dim: int = 64, freeze: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        if freeze:
            self.freeze_params()
    
    def freeze_params(self):
        """Freeze parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Project features to embedding space"""
        with torch.set_grad_enabled(not self.freeze):
            return self.layers(x)


class ModelFactory:
    """Factory for creating different backbones"""
    
    @staticmethod
    def create_backbone(config, backbone_name: str) -> Backbone:
        """Create a backbone model"""
        # Check for synthetic MLP backbone
        if backbone_name == "mlp" or backbone_name == "mlp_contrastive":
            return MLPBackbone(
                input_dim=getattr(config.experiment if hasattr(config, 'experiment') else config, 'input_dim', 64),
                hidden_dim=getattr(config.experiment if hasattr(config, 'experiment') else config, 'hidden_dim', 128),
                embedding_dim=config.model.embedding_dim,
                freeze=config.model.freeze_backbone
            )
        
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
