from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Dataset


class CIFAR100SeenUnseenSplit(Dataset):
    """CIFAR-100 dataset with seen/unseen class splits."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        seen_classes: list[int] | None = None,
        unseen_classes: list[int] | None = None,
        download: bool = True,
        transform=None,
    ):
        if seen_classes is None and unseen_classes is None:
            raise ValueError("Either seen_classes or unseen_classes must be specified")

        self.cifar100 = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=transform
        )

        self.transform = transform

        if seen_classes is not None:
            mask = np.isin(self.cifar100.targets, seen_classes)
            self.seen_classes = seen_classes
            self.unseen_classes = []
        elif unseen_classes is not None:
            mask = np.isin(self.cifar100.targets, unseen_classes)
            self.seen_classes = []
            self.unseen_classes = unseen_classes

        self.mask = mask
        self.data = self.cifar100.data[mask]
        self.targets = [
            self._remap_label(label, seen_classes, unseen_classes)
            for label in np.array(self.cifar100.targets)[mask]
        ]

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(
                self.seen_classes if seen_classes else self.unseen_classes
            )
        }

    def _remap_label(self, label: int, seen: list[int] | None, unseen: list[int] | None) -> int:
        if seen is not None and label in seen:
            return int(np.where(np.array(seen) == label)[0][0])
        if unseen is not None and label in unseen:
            return int(np.where(np.array(unseen) == label)[0][0])
        return -1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def load_cifar100_splits(
    root: str | Path = "~/data",
    batch_size: int = 64,
    num_workers: int = 4,
    seen_classes: list[int] | None = None,
    unseen_classes: list[int] | None = None,
    augment: bool = True,
) -> dict[str, DataLoader]:
    """Load CIFAR-100 with seen/unseen class splits."""

    root = Path(root).expanduser()

    if seen_classes is None:
        seen_classes = list(range(80))

    if unseen_classes is None:
        unseen_classes = list(range(80, 100))

    if augment:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(32),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_seen = CIFAR100SeenUnseenSplit(
            root=tmpdir,
            train=True,
            seen_classes=seen_classes,
            download=True,
            transform=transform,
        )
        val_seen = CIFAR100SeenUnseenSplit(
            root=tmpdir,
            train=True,
            seen_classes=seen_classes,
            download=True,
            transform=transform,
        )
        test_seen = CIFAR100SeenUnseenSplit(
            root=tmpdir,
            train=False,
            seen_classes=seen_classes,
            download=True,
            transform=transform,
        )
        test_unseen = CIFAR100SeenUnseenSplit(
            root=tmpdir,
            train=False,
            unseen_classes=unseen_classes,
            download=True,
            transform=transform,
        )

    return {
        "train_seen": DataLoader(
            train_seen, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val_seen": DataLoader(
            val_seen, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test_seen": DataLoader(
            test_seen, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test_unseen": DataLoader(
            test_unseen, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }


def get_cifar100_class_counts() -> dict[str, int]:
    """Return number of images per class for each split."""
    return {
        "train_seen": 400,
        "val_seen": 100,
        "test_seen": 100,
        "test_unseen": 100,
    }
