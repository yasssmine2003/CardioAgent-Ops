"""Tests de l'export ONNX : le modèle ONNX doit donner les mêmes probabilités que l'original."""
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_pytorch_export_matches_original(tmp_path):
    """CNN (non entraîné) exporté en ONNX : mêmes probabilités qu'avec PyTorch."""
    torch = pytest.importorskip("torch")
    from src.export_onnx import export_pytorch, onnx_predict
    from src.features import MEL_SHAPE
    from src.models import cnn

    model = cnn.build_model(dropout=0.3).eval()
    path = tmp_path / "cnn.onnx"
    io = export_pytorch(model, path, (1, *MEL_SHAPE))
    X = np.random.default_rng(0).normal(size=(8, 1, *MEL_SHAPE)).astype(np.float32)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    assert np.abs(onnx_predict(path, X, io) - expected).max() < 1e-4


def test_sklearn_export_matches_original(tmp_path):
    """SVM (Pipeline avec standardisation) exporté en ONNX : mêmes probabilités."""
    pytest.importorskip("skl2onnx")
    from src.export_onnx import export_sklearn, onnx_predict
    from src.models import svm

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 28)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = svm.build_model(C=1.0, gamma="scale").fit(X, y)
    path = tmp_path / "svm.onnx"
    io = export_sklearn(model, path, 28)
    assert np.abs(onnx_predict(path, X, io) - model.predict_proba(X)[:, 1]).max() < 1e-3


def test_deploy_threshold_is_chosen_per_recording():
    """Le seuil est appliqué à la moyenne des segments de chaque enregistrement."""
    import pandas as pd
    from src.export_onnx import choose_deploy_threshold

    meta = pd.DataFrame({"path": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
                         "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]})
    proba = np.array([0.9, 0.7, 0.6, 0.8, 0.1, 0.3, 0.2, 0.2, 0.4, 0.5])
    threshold = choose_deploy_threshold(meta, proba, min_specificity=0.8)
    assert 0.45 < threshold <= 0.7   # sépare les malades (0,8 et 0,7) des sains (≤ 0,45)