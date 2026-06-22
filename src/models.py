"""Model definitions.

Provides:
- build_model(name, num_classes=2, pretrained=True)
- build_encoder(name, pretrained=False) for SimCLR.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as models


def _build_resnet(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported ResNet variant: {name}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_vit(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "vit_b_16":
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        except AttributeError:
            raise ValueError("vit_b_16 not available in this torchvision version")
    else:
        raise ValueError(f"Unsupported ViT variant: {name}")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 2, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    """Build a classifier model.

    Parameters
    ----------
    name: str
        One of: 'resnet18', 'resnet34', 'vit_b_16' (if available).
    num_classes: int
        Number of output classes (2 for real vs fake).
    pretrained: bool
        If True, use ImageNet-pretrained weights when available.
    """
    name = name.lower()
    if name in {"resnet18", "resnet34"}:
        return _build_resnet(name, num_classes=num_classes, pretrained=pretrained)
    if name.startswith("vit"):
        return _build_vit(name, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}")


def build_encoder(name: str, pretrained: bool = False) -> nn.Module:
    """Build an encoder backbone for SimCLR.

    For ResNets, this returns the network with the final classification head
    replaced by an identity layer, so the forward pass yields feature vectors.
    """
    name = name.lower()
    if name in {"resnet18", "resnet34"}:
        model = _build_resnet(name, num_classes=1000, pretrained=pretrained)
        # Replace classification head with identity to expose features
        model.fc = nn.Identity()
        return model
    if name.startswith("vit"):
        model = _build_vit(name, num_classes=1000, pretrained=pretrained)
        model.heads.head = nn.Identity()
        return model
    raise ValueError(f"Unknown encoder name: {name}")
