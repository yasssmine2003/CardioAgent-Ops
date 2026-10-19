"""Modèle 5 — Transformer-CNN hybride sur Mel-spectrogrammes.

Architecture légère INSPIRÉE d'AST et du Conformer (ce n'est ni un AST ni un Conformer complet) :
- un front-end CNN extrait des motifs locaux et transforme l'image en séquence d'instants ;
- un encodeur Transformer compare chaque instant à tous les autres (structure temporelle B1/B2) ;
- comme dans AST : un token [CLS] résume la séquence, avec des plongements de position appris.
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import N_MELS
from src.features import N_FRAMES

PARAM_GRID = [
    {"learning_rate": 1e-3, "dropout": 0.1},
    {"learning_rate": 3e-4, "dropout": 0.1},
]

D_MODEL = 64        # taille du vecteur qui décrit chaque instant
N_HEADS = 4         # 4 « têtes » d'attention regardent la séquence en parallèle
N_LAYERS = 2        # 2 couches Transformer empilées
FF_DIM = 128        # taille de la couche intermédiaire de chaque couche Transformer


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class TransformerCNN(nn.Module):
    """Front-end CNN + encodeur Transformer + token [CLS].

    Formes (lot de B segments) :
        entrée                (B, 1, 32, 157)
        front-end CNN ->      (B, 64, 8, 39)       64 motifs x 8 bandes x 39 instants
        séquence ->           (B, 39, 512)         un vecteur par instant (64 motifs x 8 bandes)
        projection ->         (B, 39, 64)
        + [CLS] + position -> (B, 40, 64)
        Transformer ->        (B, 40, 64)          chaque instant « regarde » tous les autres
        [CLS] -> tête ->      (B,)                 un score par segment
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.frontend = nn.Sequential(conv_block(1, 32), conv_block(32, 64))
        n_freq, n_steps = N_MELS // 4, N_FRAMES // 4          # 8 bandes, 39 instants
        self.projection = nn.Linear(64 * n_freq, D_MODEL)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.position = nn.Parameter(torch.randn(1, n_steps + 1, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Dropout(dropout), nn.Linear(D_MODEL, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.frontend(x)                                   # (B, 64, 8, 39)
        b, c, f, t = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, 39, 512)
        tokens = self.projection(sequence)                            # (B, 39, 64)
        cls = self.cls_token.expand(b, -1, -1)                        # (B, 1, 64)
        tokens = torch.cat([cls, tokens], dim=1) + self.position      # (B, 40, 64)
        encoded = self.encoder(tokens)                                # (B, 40, 64)
        return self.head(encoded[:, 0]).squeeze(1)                    # le [CLS] résume tout


def build_model(dropout: float, **_) -> nn.Module:
    """Construit le Transformer-CNN (le learning rate est utilisé par l'entraînement)."""
    return TransformerCNN(dropout=dropout)