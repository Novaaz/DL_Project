"""SimCLR self-supervised pretraining.

Exposes:
    pretrain_ssl(config)  ->  str   path to saved encoder checkpoint

Config keys:
    data_root       str   path containing 'AI' and 'Human' folders
    encoder_name    str   'resnet18' (default) or 'resnet34'
    projection_dim  int   projection head output dim  (default 128)
    temperature     float NT-Xent temperature          (default 0.07)
    batch_size      int   (default 256)
    num_epochs      int   (default 100)
    lr              float (default 3e-4)
    image_size      int   (default 224)
    num_workers     int   (default 4)
    checkpoint_path str   where to save encoder weights
    device          str   'cuda' or 'cpu'
    seed            int   (default 42)

Example:
    from src.ssl import pretrain_ssl
    path = pretrain_ssl({"data_root": "data/raw", "num_epochs": 50})
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.models import build_encoder
from src.augmentations import get_train_transform
from src.utils import set_seed


# ---------------------------------------------------------------------------
# SimCLR augmentation  (used by notebooks directly)
# ---------------------------------------------------------------------------

def get_simclr_transform(image_size: int = 224) -> T.Compose:
    """Standard SimCLR augmentation (Chen et al., ICML 2020).

    Excludes gaussian_noise: nb03 shows it destroys the
    discriminative signal entirely (AUC drops to 0.795).
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([
            T.ColorJitter(brightness=0.8, contrast=0.8,
                          saturation=0.8, hue=0.2)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# SimCLR Dataset: returns two random views of the same image
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """Wraps any ImageFolder-style dataset; returns (view1, view2) pairs."""

    def __init__(self, root: str, transform) -> None:
        self._base = ImageFolder(root=root)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self._base[idx]          # PIL image, label ignored
        return self._transform(img), self._transform(img)


class UnlabeledPairDataset(Dataset):
    """Returns (view_i, view_j): two augmented views of the same image.

    Scans a directory recursively for images (no label subdirs needed).
    Used by notebooks when the data isn't in ImageFolder layout.
    """

    _EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

    def __init__(self, root: str | Path, transform: T.Compose):
        root = Path(root)
        self.paths = sorted(
            p for p in root.rglob('*')
            if p.suffix.lower() in self._EXTENSIONS
        )
        self.transform = transform
        if not self.paths:
            raise FileNotFoundError(f'No images found in {root}')
        print(f'UnlabeledPairDataset: {len(self.paths)} images from {root}')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# SimCLR model: encoder + projection head
# ---------------------------------------------------------------------------

class SimCLR(nn.Module):
    """
    f(·) = ResNet18 backbone stripped of its FC layer  →  h  (B, 512)
    g(·) = 2-layer MLP projection head                 →  z  (B, proj_dim)

    At inference / fine-tuning: discard g(·), use h = f(x) directly.
    """

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        backbone = build_encoder("resnet18", pretrained=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.feat_dim = backbone(dummy).shape[1]
        self.encoder = backbone

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)    # representation  (used after training)
        z = self.projector(h)  # projection      (used during training only)
        return h, z


# ---------------------------------------------------------------------------
# Projection head (standalone, used by pretrain_ssl)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP: backbone_dim -> 256 -> projection_dim."""

    def __init__(self, in_dim: int = 512, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NT-Xent loss (class version, used by notebooks)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """NT-Xent loss as in Chen et al. (ICML 2020).

    Temperature tau controls hardness of negatives.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)          # (2N, D)
        z = F.normalize(z, dim=1)                  # unit sphere

        sim = torch.mm(z, z.T) / self.tau           # (2N, 2N) cosine sim

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# NT-Xent loss (functional, used by pretrain_ssl)
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalised Temperature-scaled Cross-Entropy loss (SimCLR, Chen et al. 2020).

    Each pair (z1[i], z2[i]) is a positive pair; all other N-1 pairs in the
    batch are negatives.  We concatenate both views so the effective batch
    size is 2N and compute cross-entropy over the similarity matrix.

    Args:
        z1, z2: L2-normalised projections, shape (N, proj_dim)
        temperature: scaling factor tau

    Returns:
        Scalar loss averaged over both views.
    """
    N = z1.size(0)
    # Stack to (2N, proj_dim) and compute (2N x 2N) cosine similarity matrix
    z = torch.cat([z1, z2], dim=0)                           # (2N, d)
    sim = torch.mm(z, z.T) / temperature                     # (2N, 2N)

    # Mask the diagonal (self-similarity) by setting it to a large negative
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Positive indices: for i in [0,N) the positive is i+N, and vice versa
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pretrain_ssl(config: dict) -> str:
    """Run SimCLR pretraining and save the encoder backbone.

    Args:
        config: dictionary with keys described in the module docstring.

    Returns:
        Path to the saved encoder checkpoint (.pt file).
    """
    cfg = {
        "encoder_name":    "resnet18",
        "projection_dim":  128,
        "temperature":     0.07,
        "batch_size":      256,
        "num_epochs":      100,
        "lr":              3e-4,
        "image_size":      224,
        "num_workers":     4,
        "checkpoint_path": "checkpoints/simclr_encoder.pt",
        "device":          "cuda" if torch.cuda.is_available() else "cpu",
        "seed":            42,
        **config,
    }

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # Dataset & loader
    transform  = get_train_transform("simclr", cfg["image_size"])
    dataset    = SimCLRDataset(root=cfg["data_root"], transform=transform)
    loader     = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,   # NT-Xent requires full batches
    )

    # Model — use ImageNet-pretrained weights (standard SimCLR practice)
    encoder = build_encoder(cfg["encoder_name"], pretrained=True).to(device)
    # Infer backbone output dim from a dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg["image_size"], cfg["image_size"]).to(device)
        feat_dim = encoder(dummy).shape[1]

    head      = ProjectionHead(in_dim=feat_dim, proj_dim=cfg["projection_dim"]).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg["lr"],
    )

    # Training loop
    encoder.train()
    head.train()
    for epoch in range(1, cfg["num_epochs"] + 1):
        total_loss = 0.0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)

            h1, h2 = encoder(view1), encoder(view2)   # backbone features
            z1 = F.normalize(head(h1), dim=1)         # L2-normalise projections
            z2 = F.normalize(head(h2), dim=1)

            loss = nt_xent_loss(z1, z2, cfg["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"[SimCLR] epoch {epoch}/{cfg['num_epochs']}  loss={avg:.4f}")

    # Save only the encoder (projection head is discarded after pretraining)
    os.makedirs(os.path.dirname(cfg["checkpoint_path"]) or ".", exist_ok=True)
    torch.save(encoder.state_dict(), cfg["checkpoint_path"])
    print(f"[SimCLR] encoder saved to {cfg['checkpoint_path']}")
    return cfg["checkpoint_path"]
