from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class DataConfig:
    dataset: str = "cifar100"
    num_classes: int = 100
    seen_classes: int = 80
    unseen_classes: int = 20
    train_seen_ratio: float = 0.8
    seed: int = 42
    num_splits: int = 3


@dataclass
class AugmentationConfig:
    random_resized_crop: dict = field(default_factory=lambda: {
        "scale": [0.2, 1.0],
        "ratio": [0.75, 1.3333333333333333]
    })
    color_jitter: dict = field(default_factory=lambda: {
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.4,
        "hue": 0.1
    })
    random_horizontal_flip: float = 0.5


@dataclass
class LossConfig:
    contrastive: dict = field(default_factory=lambda: {
        "margin": 1.0,
        "distance_metric": "cosine"
    })
    triplet: dict = field(default_factory=lambda: {
        "margin": 0.3,
        "miner": "semi_hard"
    })
    shadow: dict = field(default_factory=lambda: {
        "projection_dim": 1,
        "learnable_projection": True
    })


@dataclass
class ModelConfig:
    backbones: List[str] = field(default_factory=lambda: [
        "resnet18", "resnet101", "vit_base_patch16_224", "clip_vit_base_patch32", "dino_vit_base_patch8"
    ])
    pretrained: bool = True
    freeze_backbone: bool = False
    embedding_dim: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_batch_size: int = 1024
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


@dataclass
class HPOConfig:
    method: str = "optuna"
    n_trials: int = 100
    timeout: int = 3600
    search_space: dict = field(default_factory=lambda: {
        "batch_size": [64, 128, 256, 512, 1024],
        "learning_rate": [1e-5, 1e-3],
        "margin": [0.1, 1.0]
    })


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "grouped_recall_at_k", "opis", "ari", "ami", "nmi", "silhouette"
    ])
    k: int = 5
    threshold_range: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    num_groups: int = 4
    statistical_test: str = "5x2cv"


@dataclass
class L2ANCConfig:
    latent_dim: int = 128
    generator_hidden_dim: int = 256
    generator_layers: int = 3
    synthetic_classes_per_batch: int = 4
    kl_divergence_weight: float = 1.0
    num_synthetic_samples_per_class: int = 100


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    l2anc: L2ANCConfig = field(default_factory=L2ANCConfig)
    project_dir: str = "/Users/glasslab/dml-project"
    wandb_enabled: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        def build_dataclass(obj_class: type, data_dict: dict) -> any:
            if not hasattr(obj_class, "__dataclass_fields__"):
                return data_dict
            dataclass_fields = obj_class.__dataclass_fields__
            kwargs = {}
            for field_name, field_info in dataclass_fields.items():
                if field_name not in data_dict:
                    continue
                field_value = data_dict[field_name]
                field_type = field_info.type
                
                if isinstance(field_value, dict):
                    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                        item_type = field_type.__args__[0]
                        kwargs[field_name] = [build_dataclass(item_type, item) for item in field_value]
                    elif hasattr(field_type, "__dataclass_fields__"):
                        kwargs[field_name] = build_dataclass(field_type, field_value)
                    else:
                        kwargs[field_name] = field_value
                else:
                    kwargs[field_name] = field_value
            return obj_class(**kwargs)
        
        return build_dataclass(cls, data)
    
    def to_dict(self) -> dict:
        return asdict(self)


def get_config(path: Optional[str] = None) -> Config:
    if path and os.path.exists(path):
        return Config.from_yaml(path)
    return Config()
