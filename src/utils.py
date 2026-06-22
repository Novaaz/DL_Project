"""Utility functions: seeds, metrics, logging, saving results, Grad-CAM."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute classification metrics for binary classification.

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, precision, recall, f1,
    specificity, auc, avg_precision, mcc, tn, fp, fn, tp
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    try:
        auc   = roc_auc_score(y_true, y_prob)
        avg_p = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = avg_p = float("nan")

    return {
        "accuracy":          float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision":         float(precision),
        "recall":            float(recall),
        "f1":                float(f1),
        "specificity":       float(specificity),
        "auc":               float(auc),
        "avg_precision":     float(avg_p),
        "mcc":               float(mcc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compute_metrics_per_class(
    y_true, y_pred, class_names: List[str] = None
) -> pd.DataFrame:
    """Per-class precision/recall/f1/support as a DataFrame."""
    from sklearn.metrics import classification_report
    if class_names is None:
        class_names = ["Human (Real)", "AI (Fake)"]
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = [{"class": cls, **report[cls]} for cls in class_names]
    rows.append({"class": "macro avg",    **report["macro avg"]})
    rows.append({"class": "weighted avg", **report["weighted avg"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def fig_path(reports_dir: str, notebook: str, filename: str) -> str:
    """Build a figure path organised by notebook subfolder.

    Convention
    ----------
    reports/
      figures/
        01_dataset_exploration/
          class_balance.png
        02_supervised_baseline/
          confusion_matrix_weak.png
        ...

    Parameters
    ----------
    reports_dir : base reports directory, e.g. '../reports'
    notebook    : notebook prefix/name, e.g. '01_dataset_exploration'
    filename    : figure filename, e.g. 'class_balance.png'
    """
    return str(Path(reports_dir) / "figures" / notebook / filename)


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, show: bool = True) -> None:
    """Save a matplotlib figure to disk and optionally display it inline.

    Parameters
    ----------
    fig  : matplotlib Figure
    path : destination path (parent dirs created automatically)
    dpi  : resolution for the saved file
    show : if True, calls plt.show() so the figure appears in the notebook cell
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log(f"Figure saved \u2192 {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    y_true, y_pred,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Raw counts + row-normalised confusion matrix side by side."""
    if class_names is None:
        class_names = ["Human (Real)", "AI (Fake)"]
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ["d", ".2%"], ["Counts", "Row-normalised"]
    ):
        ConfusionMatrixDisplay(confusion_matrix=data, display_labels=class_names).plot(
            ax=ax, colorbar=False, values_format=fmt
        )
        ax.set_title(subtitle)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path:
        save_figure(fig, path, show=show)
    elif show:
        plt.show()
    return fig


def plot_roc_curve(
    y_true, y_prob,
    label: str = "",
    path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout()
    if path:
        save_figure(fig, path, show=show)
    elif show:
        plt.show()
    return fig


def plot_pr_curve(
    y_true, y_prob,
    label: str = "",
    path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Precision-Recall curve with AP annotation."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"{label} AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve"); ax.legend()
    fig.tight_layout()
    if path:
        save_figure(fig, path, show=show)
    elif show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def save_results_csv(records: List[Dict], path: str) -> None:
    """Save a list of metric dicts to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    log(f"Results saved \u2192 {path}")


def plot_training_curves(
    history: List[Dict],
    path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot train loss and val accuracy from training history."""
    epochs     = [h["epoch"]        for h in history]
    train_loss = [h["train_loss"]   for h in history]
    val_acc    = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, marker="o")
    ax1.set_title("Train Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.plot(epochs, val_acc, marker="o", color="orange")
    ax2.set_title("Val Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    fig.tight_layout()
    if path:
        save_figure(fig, path, show=show)
    elif show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(message: str, filepath: Optional[str] = None) -> None:
    """Timestamped log to stdout and optional file."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s - %(message)s")
    _logging.info(message)
    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(message + "\n")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class _GradCAMHook:
    def __init__(self):
        self.activation = None
        self.gradient   = None

    def forward_hook(self, module, inp, out):
        self.activation = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()


def compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model        : trained classifier in eval mode
    image_tensor : (1, C, H, W) on the same device as model
    target_class : 0=real, 1=AI
    target_layer : conv layer to hook (e.g. model.layer4[-1])

    Returns
    -------
    heatmap : (H, W) np.ndarray in [0, 1]
    """
    hook = _GradCAMHook()
    fwd = target_layer.register_forward_hook(hook.forward_hook)
    bwd = target_layer.register_full_backward_hook(hook.backward_hook)

    model.eval()
    # Ensure input requires grad for backward pass
    if not image_tensor.requires_grad:
        image_tensor = image_tensor.detach().requires_grad_(True)
    out = model(image_tensor)
    model.zero_grad()
    out[0, target_class].backward()
    fwd.remove(); bwd.remove()

    weights = hook.gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * hook.activation).sum(dim=1).squeeze(0)
    cam = torch.clamp(cam, min=0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.cpu().numpy()
