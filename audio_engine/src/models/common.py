"""Briques communes aux modèles de deep learning."""
from __future__ import annotations

import torch
from torch import nn


class NormalizedModel(nn.Module):
    """Enveloppe un réseau et standardise son entrée : (x - moyenne) / écart-type.

    La moyenne et l'écart-type sont calculés sur le TRAIN et stockés DANS le modèle
    (buffers) : ils seront exportés avec lui en ONNX et appliqués automatiquement
    à l'inférence (même principe que le Pipeline du SVM).
    """

    def __init__(self, backbone: nn.Module, mean: float, std: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Nombre de paramètres du modèle (ou seulement de ceux qui sont entraînés)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)