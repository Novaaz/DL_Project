"""Semi-supervised learning: pseudo-labeling implementation.

This module implements iterative pseudo-labeling to combine a small labeled subset
with a larger unlabeled pool for improved classification performance.

Main entry point:
    train_pseudo_labeling(config) -> dict

Config keys:
    data_root            str   path containing 'AI' and 'Human' folders
    model_name           str   'resnet18' (default)
    labeled_ratio        float fraction of train data kept labeled  (default 0.1)
    confidence_threshold float min softmax prob to accept pseudo-label (default 0.95)
    num_rounds           int   pseudo-labeling rounds               (default 3)
    epochs_per_round     int   supervised epochs per round          (default 10)
    batch_size           int   (default 64)
    image_size           int   (default 224)
    num_workers          int   (default 4)
    seed                 int   (default 42)
    lr                   float (default 1e-4)
    device               str   'cuda' or 'cpu' (auto-detect)
    checkpoint_dir       str   directory for checkpoints ('checkpoints')
    augmentation_name    str   'weak' (default)

Example:
    from src.semi_supervised.pseudo_labeling import train_pseudo_labeling
    results = train_pseudo_labeling({
        "data_root": "data/raw",
        "labeled_ratio": 0.1,
        "num_rounds": 3,
        "epochs_per_round": 10,
    })
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import create_dataloaders, DataConfig, ImagePathDataset
from ..models import build_model
from ..utils import set_seed, compute_metrics, log


def _run_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer=None,
    device="cpu",
) -> Tuple[Dict, List, List, List]:
    """Run one training or evaluation epoch.

    Parameters
    ----------
    model : nn.Module
        The classifier model.
    loader : DataLoader
        Input data loader.
    criterion : nn.Module
        Loss function (typically CrossEntropyLoss).
    optimizer : torch.optim.Optimizer, optional
        Optimizer for training. If None, runs in eval mode.
    device : str
        Device to use (cpu or cuda).

    Returns
    -------
    metrics : dict
        Computed metrics (accuracy, loss, f1, auc, etc.).
    all_targets : list
        True labels (empty if loader contains unlabeled samples).
    all_preds : list
        Predicted labels.
    all_probs : list
        Predicted probabilities for the positive class.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_samples = 0
    all_targets: List = []
    all_preds: List = []
    all_probs: List = []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False, desc="Epoch"):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                # Labeled data: (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                has_targets = True
            else:
                # Unlabeled data: just inputs
                inputs = batch.to(device)
                targets = None
                has_targets = False

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            if has_targets:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                all_targets.extend(targets.detach().cpu().tolist())
            else:
                # For unlabeled data, no loss is computed during this pass
                pass

            all_preds.extend(preds.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())
            num_samples += inputs.size(0)

            if is_train and has_targets:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Compute metrics only if we have targets
    if all_targets:
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = compute_metrics(all_targets, all_preds, all_probs)
        metrics["loss"] = avg_loss
    else:
        # For unlabeled data without targets, return dummy metrics
        metrics = {
            "accuracy": 0.0,
            "loss": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "avg_precision": 0.0,
            "mcc": 0.0,
            "balanced_accuracy": 0.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

    return metrics, all_targets, all_preds, all_probs


def _assign_pseudo_labels(
    model,
    unlabeled_loader: DataLoader,
    confidence_threshold: float,
    device: str,
) -> Tuple[List[int], List[int], List[float]]:
    """Assign pseudo-labels to unlabeled samples with high confidence.

    This function runs inference on the unlabeled pool and assigns pseudo-labels
    to samples where the model's softmax confidence exceeds the threshold.
    Only high-confidence predictions are retained, ensuring label quality.

    Parameters
    ----------
    model : nn.Module
        Trained classifier model.
    unlabeled_loader : DataLoader
        DataLoader with unlabeled images (single images, no labels).
    confidence_threshold : float
        Minimum confidence (softmax probability) to accept a pseudo-label.
    device : str
        Device to use (cpu or cuda).

    Returns
    -------
    pseudo_indices : list of int
        Indices of samples assigned a pseudo-label (relative to the unlabeled pool).
    pseudo_labels : list of int
        Pseudo-labels for those samples (0 or 1).
    pseudo_confidences : list of float
        Confidence scores for the pseudo-labels.
    """
    model.eval()
    pseudo_indices: List[int] = []
    pseudo_labels: List[int] = []
    pseudo_confidences: List[float] = []

    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, leave=False, desc="Assigning pseudo-labels"):
            inputs = batch.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            for conf, pred in zip(
                confidences.detach().cpu().tolist(),
                preds.detach().cpu().tolist(),
            ):
                # Only accept pseudo-label if confidence exceeds threshold
                if conf >= confidence_threshold:
                    pseudo_indices.append(sample_idx)
                    pseudo_labels.append(pred)
                    pseudo_confidences.append(conf)
                sample_idx += 1

    return pseudo_indices, pseudo_labels, pseudo_confidences


def train_pseudo_labeling(config: Union[DataConfig, Dict]) -> Dict:
    """Train a classifier using iterative pseudo-labeling.

    This implements a semi-supervised learning approach that:
    1. Trains on a labeled subset for N epochs (warm-up).
    2. Assigns pseudo-labels to high-confidence unlabeled samples.
    3. Adds pseudo-labeled samples to the training set.
    4. Repeats for K rounds.

    Parameters
    ----------
    config : dict or DataConfig
        Configuration with keys described in the module docstring.

    Returns
    -------
    dict
        Results dictionary with:
        - best_val_acc: highest validation accuracy achieved
        - test_accuracy, test_f1, test_auc: final test metrics
        - test_mcc, test_avg_precision, test_balanced_accuracy, test_specificity
        - test_metrics_full: complete metrics dict
        - rounds_history: list of dicts with per-round metrics
        - labeled_ratio_used, num_pseudo_labeled_added: tracking info
        - test_y_true, test_y_pred, test_y_prob: raw arrays for plots
        - test_paths: file paths in test-split order
    """
    if not isinstance(config, DataConfig):
        config_dict = config
        config = DataConfig(
            data_root=config_dict["data_root"],
            batch_size=config_dict.get("batch_size", 64),
            image_size=config_dict.get("image_size", 224),
            num_workers=config_dict.get("num_workers", 4),
            split_ratios=tuple(config_dict.get("split_ratios", (0.7, 0.15, 0.15))),
            seed=config_dict.get("seed", 42),
            unlabeled_ratio=config_dict.get("unlabeled_ratio", 0.5),
        )
    else:
        config_dict = vars(config)

    set_seed(config_dict.get("seed", 42))

    device = config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = config_dict.get("model_name", "resnet18")
    augmentation_name = config_dict.get("augmentation_name", "weak")
    labeled_ratio = config_dict.get("labeled_ratio", 0.1)
    confidence_threshold = config_dict.get("confidence_threshold", 0.95)
    num_rounds = config_dict.get("num_rounds", 3)
    epochs_per_round = config_dict.get("epochs_per_round", config_dict.get("num_epochs", 10))
    checkpoint_dir = config_dict.get("checkpoint_dir", "checkpoints")
    lr = config_dict.get("lr", 1e-4)

    log(f"\n{'='*70}")
    log(f"Pseudo-labeling Training")
    log(f"{'='*70}")
    log(f"model_name={model_name} | labeled_ratio={labeled_ratio}")
    log(f"confidence_threshold={confidence_threshold} | num_rounds={num_rounds}")
    log(f"epochs_per_round={epochs_per_round} | device={device}")

    # --- Data loading ---
    # Use unlabeled_ratio=0 here so create_dataloaders returns full labeled train set.
    # We split into labeled + unlabeled ourselves below using labeled_ratio.
    if labeled_ratio < 1.0:
        train_config = DataConfig(
            data_root=config_dict["data_root"],
            batch_size=config_dict.get("batch_size", 64),
            image_size=config_dict.get("image_size", 224),
            num_workers=config_dict.get("num_workers", 4),
            split_ratios=tuple(config_dict.get("split_ratios", (0.7, 0.15, 0.15))),
            seed=config_dict.get("seed", 42),
            unlabeled_ratio=0.0,  # ← get full train set, split ourselves
        )
    else:
        train_config = config

    log(f"\nCreating dataloaders | aug='{augmentation_name}'")
    loaders = create_dataloaders(train_config, augmentation_name=augmentation_name)

    train_loader, val_loader, test_loader = loaders[:3]

    # Extract test paths for later reference
    test_paths: List[str] = [path for path, _ in test_loader.dataset.samples]

    train_dataset = train_loader.dataset

    # Split the full training set into labeled + unlabeled
    if labeled_ratio < 1.0:
        log(f"\nSplitting train set: labeled_ratio={labeled_ratio}")
        rng = np.random.default_rng(config_dict.get("seed", 42))
        train_samples = list(train_dataset.samples)
        num_labeled = max(1, int(len(train_samples) * labeled_ratio))
        indices = np.arange(len(train_samples))
        rng.shuffle(indices)
        labeled_indices = set(indices[:num_labeled].tolist())

        initial_labeled_samples: List = []
        unlabeled_samples: List = []
        for i, sample in enumerate(train_samples):
            path, _ = sample
            if i in labeled_indices:
                initial_labeled_samples.append(sample)
            else:
                unlabeled_samples.append((path, None))

        train_dataset.samples = initial_labeled_samples
        unlabeled_dataset = ImagePathDataset(
            unlabeled_samples,
            transform=train_dataset.transform,
        )

        # Recreate loaders with updated datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        log(f"  Labeled: {len(train_dataset.samples)}")
        log(f"  Unlabeled: {len(unlabeled_dataset.samples)}")
    else:
        # labeled_ratio=1.0: use create_dataloaders' own unlabeled split if requested
        unlabeled_loader = loaders[3] if len(loaders) == 4 else None
        unlabeled_dataset = unlabeled_loader.dataset if unlabeled_loader else None

    if unlabeled_loader is None or len(unlabeled_dataset) == 0:
        raise RuntimeError(
            "Pseudo-labeling requires unlabeled data. "
            "Set labeled_ratio < 1.0 or unlabeled_ratio > 0 in config."
        )

    # Initialize model
    log(f"\nBuilding model '{model_name}'")
    model = build_model(model_name, num_classes=2, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    rounds_history: List[Dict] = []
    total_pseudo_labeled_added = 0

    for round_num in range(1, num_rounds + 1):
        log(f"\n{'-'*70}")
        log(f"Round {round_num}/{num_rounds}")
        log(f"{'-'*70}")

        # Train on current labeled set
        log(f"Training on labeled set ({len(train_dataset.samples)} samples)...")
        for epoch in range(1, epochs_per_round + 1):
            train_m, _, _, _ = _run_epoch(
                model, train_loader, criterion, optimizer, device=device
            )
            val_m, _, _, _ = _run_epoch(model, val_loader, criterion, None, device=device)

            if epoch % max(1, epochs_per_round // 3) == 0 or epoch == epochs_per_round:
                log(f"  Epoch {epoch}/{epochs_per_round} | loss={train_m['loss']:.4f} | "
                    f"val_acc={val_m['accuracy']:.4f}")

            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ===== PSEUDO-LABEL ASSIGNMENT =====
        # Run inference on unlabeled pool and assign high-confidence pseudo-labels
        log(f"Assigning pseudo-labels (threshold={confidence_threshold})...")
        pseudo_indices, pseudo_labels, pseudo_confidences = _assign_pseudo_labels(
            model, unlabeled_loader, confidence_threshold, device
        )

        num_pseudo_labeled = len(pseudo_indices)
        log(f"  Pseudo-labeled: {num_pseudo_labeled} / {len(unlabeled_dataset.samples)} "
            f"({100*num_pseudo_labeled/max(1, len(unlabeled_dataset.samples)):.1f}%)")

        if num_pseudo_labeled == 0:
            log("  Warning: No samples met confidence threshold. Stopping pseudo-labeling.")
            break

        # ===== ADD PSEUDO-LABELED SAMPLES TO TRAINING SET =====
        # Build lookup: pseudo_indices → pseudo_labels
        pseudo_map = dict(zip(pseudo_indices, pseudo_labels))

        unlabeled_samples_list = list(unlabeled_dataset.samples)
        pseudo_labeled_samples: List = []
        remaining_unlabeled: List = []

        for i, sample in enumerate(unlabeled_samples_list):
            if i in pseudo_map:
                path, _ = sample
                pseudo_labeled_samples.append((path, pseudo_map[i]))
            else:
                remaining_unlabeled.append(sample)

        # Create fresh Dataset objects — avoids mutating samples after loader creation
        train_dataset = ImagePathDataset(
            list(train_dataset.samples) + pseudo_labeled_samples,
            transform=train_dataset.transform,
        )
        unlabeled_dataset = ImagePathDataset(
            remaining_unlabeled,
            transform=unlabeled_dataset.transform,
        )

        total_pseudo_labeled_added += num_pseudo_labeled

        log(f"  Training set size after pseudo-labeling: {len(train_dataset)}")
        log(f"  Remaining unlabeled: {len(unlabeled_dataset)}")

        # Recreate dataloaders for next round
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        rounds_history.append({
            "round": round_num,
            "labeled_samples": len(train_dataset.samples),
            "pseudo_labeled_this_round": num_pseudo_labeled,
            "val_acc": float(val_m["accuracy"]),
        })

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save final checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    labeled_ratio_pct = int(100 * labeled_ratio)
    ckpt_name = (
        f"{model_name}_pseudo_labeled{labeled_ratio_pct}pct_"
        f"seed{config_dict.get('seed', 42)}.pt"
    )
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save(best_state or model.state_dict(), ckpt_path)
    log(f"\nBest model saved → {ckpt_path}")

    # Evaluate on test set
    log(f"\nEvaluating on test set...")
    test_m, test_y_true, test_y_pred, test_y_prob = _run_epoch(
        model, test_loader, criterion, optimizer=None, device=device
    )

    log(f"\n{'='*70}")
    log(f"Final Results (Pseudo-labeling with labeled_ratio={labeled_ratio})")
    log(f"{'='*70}")
    log(f"Test Accuracy: {test_m['accuracy']:.4f}")
    log(f"Test F1:       {test_m['f1']:.4f}")
    log(f"Test AUC:      {test_m['auc']:.4f}")
    log(f"Best Val Acc:  {best_val_acc:.4f}")
    log(f"Total pseudo-labeled samples added: {total_pseudo_labeled_added}")
    log(f"{'='*70}\n")

    return {
        # summary scalars
        "best_val_acc": float(best_val_acc),
        "test_accuracy": float(test_m["accuracy"]),
        "test_f1": float(test_m["f1"]),
        "test_auc": float(test_m["auc"]),
        "test_mcc": float(test_m["mcc"]),
        "test_avg_precision": float(test_m["avg_precision"]),
        "test_balanced_accuracy": float(test_m["balanced_accuracy"]),
        "test_specificity": float(test_m["specificity"]),
        "test_metrics_full": test_m,
        # tracking info
        "labeled_ratio_used": float(labeled_ratio),
        "num_pseudo_labeled_added": int(total_pseudo_labeled_added),
        "rounds_history": rounds_history,
        # raw arrays for plots
        "test_y_true": test_y_true,
        "test_y_pred": test_y_pred,
        "test_y_prob": test_y_prob,
        "test_paths": test_paths,
    }
