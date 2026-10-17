"""Tests des modèles de deep learning : formes d'entrée et de sortie (sans dataset, sur CPU)."""
import pytest

torch = pytest.importorskip("torch")  # tests ignorés si PyTorch n'est pas installé

from src.features import MEL_SHAPE
from src.models import cnn
from src.models.common import NormalizedModel


def test_cnn_output_shape():
    """Un lot de 4 Mel-spectrogrammes -> 4 scores, un par segment."""
    model = cnn.build_model(dropout=0.3).eval()
    out = model(torch.randn(4, 1, *MEL_SHAPE))
    assert out.shape == (4,)


def test_normalized_model_standardizes_input():
    """Le modèle enveloppé applique (x - moyenne) / écart-type avant le réseau."""
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x.mean(dim=(1, 2, 3))

    model = NormalizedModel(Spy(), mean=-20.0, std=10.0)
    model(torch.full((2, 1, *MEL_SHAPE), -10.0))
    assert torch.allclose(seen["x"], torch.ones_like(seen["x"]))