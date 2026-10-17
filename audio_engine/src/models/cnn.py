"""Modèle 3 — CNN Custom 2D sur Mel-spectrogrammes (32 x 157)."""
from __future__ import annotations

import torch
from torch import nn

# Combinaisons testées (chacune = un run MLflow enfant)
PARAM_GRID = [
    {"learning_rate": 1e-3, "dropout": 0.3},
    {"learning_rate": 3e-4, "dropout": 0.3},
]


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 : détecte des motifs puis réduit l'image de moitié."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNNCustom(nn.Module):
    """Petit CNN : 3 blocs de convolution, moyenne globale, une couche de décision.

    Formes (pour un lot de B segments) :
        entrée        (B, 1, 32, 157)
        bloc 1 ->     (B, 16, 16, 78)
        bloc 2 ->     (B, 32,  8, 39)
        bloc 3 ->     (B, 64,  4, 19)
        moyenne ->    (B, 64)
        sortie ->     (B,)   un score (logit) par segment ; sigmoïde -> probabilité
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16), conv_block(16, 32), conv_block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x))).squeeze(1)


def build_model(dropout: float, **_) -> nn.Module:
    """Construit le réseau (le learning rate est utilisé par l'entraînement, pas ici)."""
    return CNNCustom(dropout=dropout)