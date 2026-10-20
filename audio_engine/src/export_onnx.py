"""Export du modèle sélectionné au format ONNX et choix du seuil de déploiement.

1. charge le modèle choisi au benchmark (data/reports/best_model.json) depuis MLflow ;
2. l'exporte en ONNX (PyTorch -> torch.onnx ; scikit-learn -> skl2onnx) ;
3. vérifie qu'ONNX Runtime donne les mêmes probabilités que le modèle d'origine ;
4. choisit le seuil de déploiement PAR ENREGISTREMENT sur la validation
   (spécificité >= DEPLOY_MIN_SPECIFICITY), avec les probabilités du modèle ONNX ;
5. écrit la fiche du modèle (cardio_model.json) : seuil, prétraitement, résultats.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort

from src import config
from src.benchmark import LoadedModel, load_inputs, predict_proba, tune_threshold
from src.evaluation import aggregate_by_recording, compute_metrics

logger = logging.getLogger(__name__)
# Écart maximal toléré entre le modèle d'origine et le modèle ONNX (scikit-learn calcule en
# double précision, ONNX en simple précision : on tolère un écart un peu plus grand)
MAX_ABS_DIFF = {"deep": 1e-4, "classical": 1e-3}


# ============================ 1. Chargement ============================

def load_selection(path: Path = config.BEST_MODEL_PATH) -> dict:
    """Modèle choisi au benchmark (identifiant MLflow, famille, seuil du benchmark...)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord `python -m scripts.benchmark`")
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_model(selection: dict) -> LoadedModel:
    """Recharge depuis MLflow exactement le run sélectionné (sur CPU)."""
    if selection["family"] == "classical":
        model = mlflow.sklearn.load_model(selection["model_uri"])
    else:
        import torch
        model = mlflow.pytorch.load_model(selection["model_uri"], map_location=torch.device("cpu")).eval()
    return LoadedModel(selection["model_name"], selection["family"], selection["run_id"], model, {})


# ============================== 2. Export ==============================

def export_pytorch(model, path: Path, input_shape: tuple[int, ...]) -> dict:
    """Exporte un réseau PyTorch. La sortie ONNX est directement la probabilité (sigmoïde incluse)."""
    import torch
    from torch import nn

    class ProbabilityModel(nn.Module):
        def __init__(self, net: nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.net(x))

    wrapper = ProbabilityModel(model).eval()
    dummy = torch.zeros(1, *input_shape)
    torch.onnx.export(wrapper, dummy, str(path), input_names=["mel_spectrogram"],
                      output_names=["probability"], opset_version=config.ONNX_OPSET,
                      dynamic_axes={"mel_spectrogram": {0: "batch"}, "probability": {0: "batch"}},
                      dynamo=False)
    return {"input_name": "mel_spectrogram", "input_shape": ["batch", *input_shape],
            "output_name": "probability", "output_kind": "probability_pathological"}


def export_sklearn(model, path: Path, n_features: int) -> dict:
    """Exporte un modèle scikit-learn (Pipeline compris). Nécessite le paquet skl2onnx."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, n_features]))],
                                 options={id(model): {"zipmap": False}}, target_opset=config.ONNX_OPSET)
    path.write_bytes(onnx_model.SerializeToString())
    return {"input_name": "features", "input_shape": ["batch", n_features],
            "output_name": "probabilities", "output_kind": "class_probabilities"}


def onnx_predict(path: Path, X: np.ndarray, io: dict, batch_size: int = 512) -> np.ndarray:
    """Probabilité « pathologique » calculée par ONNX Runtime (comme le fera le service)."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = [session.run([io["output_name"]], {io["input_name"]: X[i:i + batch_size].astype(np.float32)})[0]
               for i in range(0, len(X), batch_size)]
    proba = np.concatenate(outputs)
    return proba[:, 1] if io["output_kind"] == "class_probabilities" else proba


# ======================= 3. Seuil de déploiement =======================

def choose_deploy_threshold(meta_val, p_val: np.ndarray,
                            min_specificity: float = config.DEPLOY_MIN_SPECIFICITY) -> float:
    """Seuil appliqué à la MOYENNE des probabilités des segments d'un enregistrement :
    Recall pathologique maximal avec une spécificité par enregistrement >= min_specificity."""
    labels, proba = aggregate_by_recording(meta_val, p_val)
    return tune_threshold(labels, proba, min_specificity)


# ============================== 4. Fiche ==============================

def preprocessing_card() -> dict:
    """Réglages que le service doit reproduire à l'identique avant d'appeler le modèle."""
    info = json.loads(config.FEATURES_INFO_PATH.read_text(encoding="utf-8"))
    return {
        "sample_rate": config.TARGET_SR,
        "segment_duration_s": config.SEGMENT_DURATION_S,
        "segment_hop_s": config.SEGMENT_HOP_S,
        "min_duration_s": config.MIN_DURATION_S,
        "amplitude_normalization": "z-score par enregistrement",
        "short_audio_padding": "répétition jusqu'à la durée d'un segment",
        **{k: info[k] for k in ["n_fft", "hop_length", "n_mels", "n_mfcc", "fmax", "n_frames"]},
    }


def export_best_model() -> dict:
    """Exporte le modèle sélectionné, vérifie l'export, choisit le seuil et écrit la fiche."""
    selection = load_selection()
    loaded = load_selected_model(selection)
    X_val, _, meta_val = load_inputs(loaded.family, "val")
    X_test, y_test, meta_test = load_inputs(loaded.family, "test")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ONNX_MODEL_PATH
    io = (export_pytorch(loaded.model, path, X_test.shape[1:]) if loaded.family == "deep"
          else export_sklearn(loaded.model, path, X_test.shape[1]))
    logger.info("Modèle exporté : %s (%.3f Mo)", path, path.stat().st_size / 1e6)

    # Vérification : mêmes probabilités qu'avec le modèle d'origine
    p_test = onnx_predict(path, X_test, io)
    max_diff = float(np.abs(p_test - predict_proba(loaded, X_test)).max())
    if max_diff > MAX_ABS_DIFF[loaded.family]:
        raise RuntimeError(f"Export ONNX non fidèle : écart maximal {max_diff:.2e}")
    logger.info("Vérification ONNX Runtime : écart maximal %.2e", max_diff)

    # Seuil de déploiement par enregistrement, choisi sur la validation
    p_val = onnx_predict(path, X_val, io)
    threshold = choose_deploy_threshold(meta_val, p_val)
    rec_val = compute_metrics(*aggregate_by_recording(meta_val, p_val), threshold)
    rec_test = compute_metrics(*aggregate_by_recording(meta_test, p_test), threshold)
    seg_test = compute_metrics(y_test, p_test, threshold)

    import onnx
    card = {
        "model_name": loaded.name,
        "family": loaded.family,
        "mlflow_run_id": loaded.run_id,
        "onnx_file": path.name,
        "onnx_opset": config.ONNX_OPSET,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **io,
        "decision": {
            "threshold": threshold,
            "level": "enregistrement (moyenne des probabilités des segments)",
            "rule": f"Recall pathologique maximal avec spécificité >= {config.DEPLOY_MIN_SPECIFICITY} "
                    "par enregistrement, choisi sur la validation",
            "labels": {"0": "normal", "1": "pathologique"},
        },
        "preprocessing": preprocessing_card(),
        "metrics": {
            "validation_recording": {k: rec_val[k] for k in ["recall_pathological", "specificity", "accuracy"]},
            "test_recording": {k: rec_test[k] for k in ["recall_pathological", "specificity", "accuracy", "f1", "roc_auc"]},
            "test_segment": {k: seg_test[k] for k in ["recall_pathological", "specificity", "accuracy", "f1", "roc_auc"]},
            "onnx_max_abs_diff": max_diff,
        },
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card