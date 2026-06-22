"""Dataset utilities for real vs AI-generated images.

Uses the Kaggle "AI art vs Human art" dataset structure:
- data/root/AI/
- data/root/Human/

Provides a single entry point:
- create_dataloaders(config, augmentation_name=None)

This returns train/val/test DataLoaders and optionally an unlabeled DataLoader
for semi-supervised experiments when config['unlabeled_ratio'] > 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from . import augmentations


CLASS_MAPPING = {
    "Human": 0,  # real
    "AI": 1,     # fake
}


@dataclass
class DataConfig:
    data_root: str
    batch_size: int = 64
    image_size: int = 224
    num_workers: int = 4
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42
    unlabeled_ratio: float = 0.0
    held_out_generators: Optional[List[str]] = None  # for future multi-generator extension


# ---------------------------------------------------------------------------
# DataLoader factory  (reduces boilerplate)
# ---------------------------------------------------------------------------

def _make_loader(dataset: Dataset, config: DataConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
    )


class ImagePathDataset(Dataset):
    """Simple Dataset from (image_path, label) pairs.

    label can be int (0/1) or None for unlabeled data.
    """

    def __init__(self, samples: List[Tuple[str, Optional[int]]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if label is None:
            return img
        return img, label


def _gather_samples(data_root: str) -> Tuple[List[str], List[int]]:
    """Scan data_root for AI/Human subfolders and build lists of paths and labels."""

    paths: List[str] = []
    labels: List[int] = []

    for class_name, label in CLASS_MAPPING.items():
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    paths.append(os.path.join(root, fname))
                    labels.append(label)
    if not paths:
        raise RuntimeError(f"No images found under {data_root}. Expected AI/ and Human/ subfolders.")
    return paths, labels


def _train_val_test_split(
    paths: List[str],
    labels: List[int],
    split_ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return index lists for train, val, test using stratified splits."""

    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split_ratios must sum to 1.0")

    paths = np.array(paths)
    labels = np.array(labels)

    # First split: train vs (val+test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(paths)),
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_rel = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=(1.0 - val_rel),
        stratify=y_temp,
        random_state=seed,
    )

    return list(train_idx), list(val_idx), list(test_idx)


def create_dataloaders(
    config: Union[DataConfig, Dict],
    augmentation_name: Optional[str] = None,
):
    """Create train/val/test (and optionally unlabeled) DataLoaders.

    Parameters
    ----------
    config: DataConfig or dict
        Configuration with fields described in DataConfig.
    augmentation_name: str, optional
        Name of the augmentation pipeline to use for training, defined in augmentations.py.

    Returns
    -------
    (train_loader, val_loader, test_loader)
        Always returned.
    (train_loader, val_loader, test_loader, unlabeled_loader)
        Returned when config.unlabeled_ratio > 0.
    """

    if not isinstance(config, DataConfig):
        config = DataConfig(**config)  # type: ignore[arg-type]

    paths, labels = _gather_samples(config.data_root)
    train_idx, val_idx, test_idx = _train_val_test_split(
        paths, labels, config.split_ratios, config.seed
    )

    image_size = config.image_size

    train_transform = augmentations.get_train_transform(
        augmentation_name or "weak", image_size=image_size
    )
    eval_transform = augmentations.get_eval_transform(image_size=image_size)

    # Build labeled splits
    def make_split(indices: List[int], unlabeled: bool = False):
        split_samples: List[Tuple[str, Optional[int]]] = []
        for idx in indices:
            label = None if unlabeled else int(labels[idx])
            split_samples.append((paths[idx], label))
        return split_samples

    train_samples_full = make_split(train_idx, unlabeled=False)
    val_samples = make_split(val_idx, unlabeled=False)
    test_samples = make_split(test_idx, unlabeled=False)

    unlabeled_loader = None

    # If semi-supervised is requested, carve out an unlabeled subset from the training set
    if config.unlabeled_ratio > 0.0:
        rng = np.random.default_rng(config.seed)
        num_unlabeled = int(len(train_samples_full) * config.unlabeled_ratio)
        indices = np.arange(len(train_samples_full))
        rng.shuffle(indices)
        unlabeled_indices = set(indices[:num_unlabeled].tolist())

        labeled_samples: List[Tuple[str, Optional[int]]] = []
        unlabeled_samples: List[Tuple[str, Optional[int]]] = []
        for i, (path, label) in enumerate(train_samples_full):
            if i in unlabeled_indices:
                unlabeled_samples.append((path, None))
            else:
                labeled_samples.append((path, label))
        train_samples_full = labeled_samples

        unlabeled_dataset = ImagePathDataset(unlabeled_samples, transform=train_transform)
        unlabeled_loader = _make_loader(unlabeled_dataset, config, shuffle=True)

    train_dataset = ImagePathDataset(train_samples_full, transform=train_transform)
    val_dataset = ImagePathDataset(val_samples, transform=eval_transform)
    test_dataset = ImagePathDataset(test_samples, transform=eval_transform)

    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = _make_loader(val_dataset, config, shuffle=False)
    test_loader = _make_loader(test_dataset, config, shuffle=False)

    if unlabeled_loader is not None:
        return train_loader, val_loader, test_loader, unlabeled_loader
    return train_loader, val_loader, test_loader
