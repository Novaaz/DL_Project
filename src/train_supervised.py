"""Supervised training loop for real vs AI-generated image classification."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import create_dataloaders, DataConfig
from .models import build_model
from .utils import set_seed, compute_metrics, log


def _run_epoch(
    model, loader: DataLoader, criterion, optimizer=None, device="cpu"
) -> Tuple[Dict, List, List, List]:
    """Run one epoch (train or eval).

    Returns
    -------
    metrics : dict of scalar metrics
    all_targets, all_preds, all_probs : raw lists (useful for plots)
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_targets: List = []
    all_preds:   List = []
    all_probs:   List = []

    for batch in tqdm(loader, leave=False):
        inputs, targets = batch
        inputs  = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(dim=1)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss   += loss.item() * inputs.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    avg_loss          = total_loss / len(loader.dataset)
    metrics           = compute_metrics(all_targets, all_preds, all_probs)
    metrics["loss"]   = avg_loss
    return metrics, all_targets, all_preds, all_probs


def _load_simclr_encoder(model, encoder_path: str, device: str) -> None:
    """Load SimCLR-pretrained encoder weights into the model backbone."""
    state = torch.load(encoder_path, map_location=device)
    if hasattr(model, "backbone"):
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    else:
        model_state = model.state_dict()
        filtered    = {
            k: v for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model_state.update(filtered)
        missing    = [k for k in state if k not in filtered]
        unexpected = []
        model.load_state_dict(model_state)
    log(f"SimCLR encoder loaded from {encoder_path} | missing={len(missing)} unexpected={len(unexpected)}")


def train_supervised(config: Union[DataConfig, Dict]) -> Dict:
    """Train a supervised classifier and evaluate on test set.

    Parameters
    ----------
    config : dict or DataConfig with keys:
        data_root, model_name, augmentation_name, num_epochs,
        batch_size, image_size, num_workers, checkpoint_dir,
        device, seed, pretrained_encoder_path (optional)

    Returns
    -------
    dict with keys:
        best_val_acc, test_accuracy, test_f1, test_auc, test_mcc,
        test_avg_precision, test_balanced_accuracy, test_specificity,
        test_metrics_full, history,
        test_y_true, test_y_pred, test_y_prob,   <- raw arrays for plots
        test_paths                                <- file paths in test-split order
    """
    if not isinstance(config, DataConfig):
        data_cfg = DataConfig(
            data_root      = config["data_root"],
            batch_size     = config.get("batch_size", 64),
            image_size     = config.get("image_size", 224),
            num_workers    = config.get("num_workers", 4),
            split_ratios   = tuple(config.get("split_ratios", (0.7, 0.15, 0.15))),
            seed           = config.get("seed", 42),
            unlabeled_ratio= config.get("unlabeled_ratio", 0.0),
        )
    else:
        data_cfg = config
        config   = vars(data_cfg)

    set_seed(config.get("seed", 42))

    device             = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_name         = config.get("model_name", "resnet18")
    augmentation_name  = config.get("augmentation_name", "weak")
    num_epochs         = config.get("num_epochs", 10)
    checkpoint_dir     = config.get("checkpoint_dir", "checkpoints")

    log(f"Creating dataloaders | aug='{augmentation_name}'")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_cfg, augmentation_name=augmentation_name
    )

    # ImagePathDataset stores (path, label) tuples in self.samples
    test_paths: List[str] = [path for path, _ in test_loader.dataset.samples]

    log(f"Building model '{model_name}'")
    model = build_model(model_name, num_classes=2, pretrained=True)

    pretrained_encoder_path = config.get("pretrained_encoder_path")
    if pretrained_encoder_path:
        _load_simclr_encoder(model, pretrained_encoder_path, device)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.get("lr", 1e-4))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    history: List[Dict] = []

    for epoch in range(1, num_epochs + 1):
        train_m, _, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device=device)
        val_m,   _, _, _ = _run_epoch(model, val_loader,   criterion, None,      device=device)
        log(f"Epoch {epoch}/{num_epochs} | train_loss={train_m['loss']:.4f} val_acc={val_m['accuracy']:.4f}")
        history.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_name = f"{model_name}_{augmentation_name}_seed{config.get('seed', 42)}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        torch.save(best_state, ckpt_path)
        log(f"Best model saved \u2192 {ckpt_path}")
        model.load_state_dict(best_state)

    log("Evaluating on test set")
    test_m, test_y_true, test_y_pred, test_y_prob = _run_epoch(
        model, test_loader, criterion, optimizer=None, device=device
    )

    return {
        # summary scalars
        "best_val_acc":           float(best_val_acc),
        "test_accuracy":          float(test_m["accuracy"]),
        "test_f1":                float(test_m["f1"]),
        "test_auc":               float(test_m["auc"]),
        "test_mcc":               float(test_m["mcc"]),
        "test_avg_precision":     float(test_m["avg_precision"]),
        "test_balanced_accuracy": float(test_m["balanced_accuracy"]),
        "test_specificity":       float(test_m["specificity"]),
        "test_metrics_full":      test_m,
        "history":                history,
        # raw arrays needed by notebook plots
        "test_y_true": test_y_true,
        "test_y_pred": test_y_pred,
        "test_y_prob": test_y_prob,
        "test_paths":  test_paths,
    }
