"""
Augmentation module: Robust contrastive learning transformations
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AugmentationPipeline:
    """Data augmentation pipeline for contrastive learning"""
    
    def __init__(self, config):
        self.config = config
        
    def get_train_transforms(self):
        """Get training augmentation pipeline"""
        augment_config = self.config.augmentation
        
        transforms_list = [
            transforms.RandomResizedCrop(
                224,
                scale=augment_config.random_resized_crop["scale"],
                ratio=augment_config.random_resized_crop["ratio"]
            ),
            transforms.ColorJitter(
                brightness=augment_config.color_jitter["brightness"],
                contrast=augment_config.color_jitter["contrast"],
                saturation=augment_config.color_jitter["saturation"],
                hue=augment_config.color_jitter["hue"]
            ),
            transforms.RandomHorizontalFlip(p=augment_config.random_horizontal_flip),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ]
        
        return transforms.Compose(transforms_list)
    
    def get_test_transforms(self):
        """Get test-time transformations (no augmentation)"""
        transforms_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ]
        
        return transforms.Compose(transforms_list)
