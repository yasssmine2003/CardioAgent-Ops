"""Modèle 4 — CNN pré-entraîné (Transfer Learning) : ResNet18 adapté aux Mel-spectrogrammes."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# Deux stratégies de Transfer Learning comparées (chacune = un run MLflow enfant)
PARAM_GRID = [
    # Fine-tuning complet : tout le réseau s'adapte, avec un petit learning rate
    {"learning_rate": 1e-4, "dropout": 0.3, "freeze_backbone": False},
    # Extraction de caractéristiques : le réseau pré-entraîné est figé, seule la tête apprend
    {"learning_rate": 1e-3, "dropout": 0.3, "freeze_backbone": True},
]


def adapt_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Transforme la 1re convolution (3 canaux RVB) en convolution à 1 canal.

    Les poids des 3 canaux sont additionnés : c'est exactement ce que ferait le réseau
    d'origine si on lui donnait 3 fois la même image en niveaux de gris. On garde donc
    tout ce que la couche a appris, sans tripler l'entrée.
    """
    new_conv = nn.Conv2d(1, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
    return new_conv


class ResNet18Audio(nn.Module):
    """ResNet18 pré-entraîné sur ImageNet, adapté à des images de 1 x 32 x 157.

    Formes (lot de B segments), après suppression du max-pooling initial :
        entrée          (B,   1, 32, 157)
        conv1 (pas 2) ->(B,  64, 16,  79)
        layer1 ->       (B,  64, 16,  79)
        layer2 ->       (B, 128,  8,  40)
        layer3 ->       (B, 256,  4,  20)
        layer4 ->       (B, 512,  2,  10)
        moyenne ->      (B, 512)
        tête ->         (B,)    un score par segment
    """

    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = False,
                 pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = adapt_first_conv(net.conv1)
        # Nos images sont petites (32 x 157) : on supprime le max-pooling initial pour
        # ne pas réduire l'image trop tôt (sinon il ne resterait que 1 x 5 en sortie).
        net.maxpool = nn.Identity()
        if freeze_backbone:
            for param in net.parameters():
                param.requires_grad = False
        # Nouvelle tête de décision (toujours entraînable) : 512 caractéristiques -> 1 score
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, 1))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(dropout: float, freeze_backbone: bool = False, pretrained: bool = True,
                **_) -> nn.Module:
    """Construit le ResNet18 adapté (le learning rate est utilisé par l'entraînement)."""
    return ResNet18Audio(dropout=dropout, freeze_backbone=freeze_backbone, pretrained=pretrained)