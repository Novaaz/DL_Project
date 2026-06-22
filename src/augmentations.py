"""Augmentation pipelines for training, eval, and SimCLR.

Three families:
  - eval:        fixed resize/crop/normalize  (val & test)
  - supervised:  none | weak | study singles  (train + robustness study)
  - simclr:      strong stochastic views      (SSL pretraining only)

Usage:
    from src.augmentations import get_train_transform, get_eval_transform, list_augmentation_names
    tf = get_train_transform("weak", image_size=224)
    study_names = list_augmentation_names()   # excludes "simclr"
"""
from __future__ import annotations

import io
from typing import Callable, Dict, List

import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class JPEGCompression:
    """Simulate JPEG artefacts via PIL in-memory encode/decode."""

    def __init__(self, quality: int = 30) -> None:
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    """Add Gaussian noise to a *tensor* image (apply after ToTensor)."""

    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, img):  # img: FloatTensor C×H×W
        import torch
        return img + torch.randn_like(img) * self.std


# ---------------------------------------------------------------------------
# Shared normalisation tail
# ---------------------------------------------------------------------------

def _norm(image_size: int) -> list:
    return [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def _norm_tail() -> list:
    """Shared ToTensor + Normalize tail for train pipelines."""
    return [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


# ---------------------------------------------------------------------------
# Eval transform
# ---------------------------------------------------------------------------

def get_eval_transform(image_size: int) -> T.Compose:
    """Deterministic transform for val / test sets."""
    return T.Compose(_norm(image_size))


# ---------------------------------------------------------------------------
# Supervised / study transforms
# ---------------------------------------------------------------------------

def _none(image_size: int) -> T.Compose:
    """Used as supervised baseline."""
    return T.Compose(_norm(image_size))


def _weak(image_size: int) -> T.Compose:
    """Light augmentation: random crop + horizontal flip."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        *_norm_tail(),
    ])


# -- Study singles (weak base + ONE extra transform) --

def _jpeg(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size + 32),
        JPEGCompression(quality=30),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        *_norm_tail(),
    ])


def _blur(image_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        *_norm_tail(),
    ])


def _noise(image_size: int) -> T.Compose:
    """ToTensor must come before GaussianNoise (operates on tensors)."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        GaussianNoise(std=0.05),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _color_jitter(image_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        *_norm_tail(),
    ])


def _random_crop(image_size: int) -> T.Compose:
    """More aggressive crop (scale down to 0.6) to test spatial robustness."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        *_norm_tail(),
    ])


# ---------------------------------------------------------------------------
# SimCLR views  (strong – used exclusively during SSL pretraining)
# ---------------------------------------------------------------------------

def _simclr(image_size: int) -> T.Compose:
    """Strong stochastic pipeline that produces two diverse views per image."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        *_norm_tail(),
    ])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[int], T.Compose]] = {
    # supervised / study
    "none":             _none,
    "weak":             _weak,
    "jpeg_compression": _jpeg,
    "gaussian_blur":    _blur,
    "gaussian_noise":   _noise,
    "color_jitter":     _color_jitter,
    "random_crop":      _random_crop,
    # ssl only
    "simclr":           _simclr,
}


def get_train_transform(augmentation_name: str, image_size: int) -> T.Compose:
    """Return the train transform for *augmentation_name* at *image_size*.

    Args:
        augmentation_name: one of the keys returned by list_augmentation_names()
                           plus "simclr" (used internally by pretrain_ssl).
        image_size: target square size in pixels (e.g. 224).

    Returns:
        A torchvision Compose pipeline.

    Raises:
        ValueError: if the name is not in the registry.
    """
    if augmentation_name not in _REGISTRY:
        raise ValueError(
            f"Unknown augmentation '{augmentation_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[augmentation_name](image_size)


def list_augmentation_names() -> List[str]:
    """Return names available for the robustness study (excludes 'simclr')."""
    return sorted(k for k in _REGISTRY if k != "simclr")
