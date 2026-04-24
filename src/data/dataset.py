"""
Data module: CIFAR-100 splits with Train-seen, Val-seen, Test-seen, Test-unseen
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List
from config import Config
import random


class Cifar100Splitter:
    """Split CIFAR-100 into disjoint seen/unseen sets with Train/Val/Test partitions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.seed = config.data.seed
        self.set_seed()
        
    def set_seed(self):
        """Set reproducibility seeds"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            
    def create_stratified_splits(self) -> Tuple[List[int], List[int]]:
        """Create stratified class splits for reproducibility across multiple seeds"""
        all_classes = list(range(100))
        
        seen_classes_list = []
        unseen_classes_list = []
        
        for split_idx in range(self.config.data.num_splits):
            np.random.shuffle(all_classes)
            seen = all_classes[:self.config.data.seen_classes]
            unseen = all_classes[self.config.data.seen_classes:]
            seen_classes_list.append(sorted(seen))
            unseen_classes_list.append(sorted(unseen))
        
        return seen_classes_list, unseen_classes_list
    
    def split_class_data(self, dataset: datasets.CIFAR100, 
                         class_indices: List[int]) -> Tuple[Subset, Subset]:
        """Split dataset into train (80%) and val (20%) for specific classes"""
        targets = np.array(dataset.targets)
        
        train_mask = np.zeros(len(dataset), dtype=bool)
        val_mask = np.zeros(len(dataset), dtype=bool)
        
        for class_idx in class_indices:
            class_positions = np.where(targets == class_idx)[0]
            np.random.shuffle(class_positions)
            
            n_train = int(len(class_positions) * self.config.data.train_seen_ratio)
            train_positions = class_positions[:n_train]
            val_positions = class_positions[n_train:]
            
            train_mask[train_positions] = True
            val_mask[val_positions] = True
            
        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        
        return Subset(dataset, train_indices), Subset(dataset, val_indices)
    
    def get_test_sets(self, dataset: datasets.CIFAR100,
                      seen_classes: List[int],
                      unseen_classes: List[int]) -> Tuple[Subset, Subset]:
        """Extract test sets for seen and unseen classes from official test set"""
        targets = np.array(dataset.targets)
        
        seen_mask = np.zeros(len(dataset), dtype=bool)
        unseen_mask = np.zeros(len(dataset), dtype=bool)
        
        for class_idx in seen_classes:
            seen_mask[targets == class_idx] = True
            
        for class_idx in unseen_classes:
            unseen_mask[targets == class_idx] = True
            
        seen_indices = np.where(seen_mask)[0].tolist()
        unseen_indices = np.where(unseen_mask)[0].tolist()
        
        return Subset(dataset, seen_indices), Subset(dataset, unseen_indices)
    
    def create_dataloaders(self) -> dict:
        """Create all dataloaders for the experiment"""
        augment_config = self.config.augmentation
        
        # Define augmentation pipeline
        transform_train = transforms.Compose([
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
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # Load full datasets
        full_train = datasets.CIFAR100(
            root=f"{self.config.project_dir}/data",
            train=True,
            download=True,
            transform=transform_train
        )
        
        full_test = datasets.CIFAR100(
            root=f"{self.config.project_dir}/data",
            train=False,
            download=True,
            transform=transform_test
        )
        
        # Create splits
        seen_classes_list, unseen_classes_list = self.create_stratified_splits()
        
        dataloaders = {}
        
        for i in range(self.config.data.num_splits):
            seen_classes = seen_classes_list[i]
            unseen_classes = unseen_classes_list[i]
            
            # Train and Val for seen classes
            train_seen, val_seen = self.split_class_data(full_train, seen_classes)
            
            # Test sets
            test_seen, test_unseen = self.get_test_sets(full_test, seen_classes, unseen_classes)
            
            # Create dataloaders
            batch_size = self.config.training.batch_size
            
            dataloaders[f"train_seen_{i}"] = DataLoader(
                train_seen, batch_size=batch_size, shuffle=True, num_workers=4
            )
            dataloaders[f"val_seen_{i}"] = DataLoader(
                val_seen, batch_size=batch_size, shuffle=False, num_workers=4
            )
            dataloaders[f"test_seen_{i}"] = DataLoader(
                test_seen, batch_size=batch_size, shuffle=False, num_workers=4
            )
            dataloaders[f"test_unseen_{i}"] = DataLoader(
                test_unseen, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            # Store class indices for reference
            dataloaders[f"seen_classes_{i}"] = seen_classes
            dataloaders[f"unseen_classes_{i}"] = unseen_classes
        
        return dataloaders


def get_dataloaders(config: Config) -> dict:
    """Convenience function to create all dataloaders"""
    splitter = Cifar100Splitter(config)
    return splitter.create_dataloaders()
