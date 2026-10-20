"""Inférence du Service 3 avec ONNX Runtime (sans PyTorch).

    .wav -> chargement -> 2000 Hz -> normalisation -> segments de 5 s
         -> Mel-spectrogrammes -> ONNX Runtime -> probabilité par segment
         -> moyenne par enregistrement -> seuil -> réponse validée (Pydantic)

Les fonctions de prétraitement et d'extraction sont EXACTEMENT celles de l'entraînement.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src import config
from src.features import N_MFCC, extract_features
from src.preprocessing import load_audio, preprocess_signal
from src.schemas import AudioFeatures, ModelInfo, PredictionResult

# Réglages qui doivent être identiques entre l'entraînement (fiche du modèle) et le service
CHECKED_SETTINGS = {
    "sample_rate": config.TARGET_SR, "segment_duration_s": config.SEGMENT_DURATION_S,
    "n_fft": config.N_FFT, "hop_length": config.HOP_LENGTH, "n_mels": config.N_MELS,
}


class ModelConfigError(RuntimeError):
    """La fiche du modèle ne correspond pas aux réglages du service."""


class CardioPredictor:
    """Charge une seule fois le modèle ONNX et sa fiche, puis prédit des enregistrements."""

    def __init__(self, model_path: Path = config.ONNX_MODEL_PATH,
                 card_path: Path = config.MODEL_CARD_PATH) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Modèle ONNX introuvable : {model_path}")
        self.card = json.loads(Path(card_path).read_text(encoding="utf-8"))
        self._check_settings()
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.card["input_name"]
        self.output_name = self.card["output_name"]
        self.threshold = float(self.card["decision"]["threshold"])

    def _check_settings(self) -> None:
        """Évite le piège classique : un modèle utilisé avec un prétraitement différent."""
        card = self.card["preprocessing"]
        wrong = {k: (card.get(k), v) for k, v in CHECKED_SETTINGS.items() if card.get(k) != v}
        if wrong:
            raise ModelConfigError(f"Réglages incompatibles (fiche, service) : {wrong}")

    def predict_signal(self, y: np.ndarray, sr: int) -> PredictionResult:
        """Prédit un enregistrement déjà chargé en mémoire (utilisé par l'API)."""
        start = time.perf_counter()
        segments = preprocess_signal(y, sr)
        vectors, mels = zip(*(extract_features(segment) for segment in segments))
        batch = np.stack(mels)[:, np.newaxis].astype(np.float32)        # (n, 1, 32, 157)
        segment_proba = self.session.run([self.output_name], {self.input_name: batch})[0]
        probability = float(np.mean(segment_proba))                       # décision par enregistrement
        is_pathological = probability >= self.threshold
        vectors = np.stack(vectors)
        duration_s = len(y) / sr

        return PredictionResult(
            prediction="pathologique" if is_pathological else "normal",
            probability=probability,
            confidence=probability if is_pathological else 1 - probability,
            segment_probabilities=[float(p) for p in segment_proba],
            n_segments=len(segments),
            duration_s=round(duration_s, 3),
            original_sample_rate=sr,
            padded=duration_s < config.SEGMENT_DURATION_S,
            features=AudioFeatures(mfcc_mean=vectors[:, :N_MFCC].mean(axis=0).tolist(),
                                   zcr_mean=float(vectors[:, 2 * N_MFCC].mean())),
            model=ModelInfo(name=self.card["model_name"], mlflow_run_id=self.card["mlflow_run_id"],
                            threshold=self.threshold, decision_level=self.card["decision"]["level"]),
            sample_rate=config.TARGET_SR,
            segment_duration_s=config.SEGMENT_DURATION_S,
            inference_time_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def predict(self, audio_path: str | Path) -> PredictionResult:
        """Prédit un fichier .wav."""
        y, sr = load_audio(audio_path)
        return self.predict_signal(y, sr)


def predict(audio_path: str | Path) -> dict:
    """Raccourci : prédit un fichier avec le modèle par défaut et renvoie un dictionnaire JSON."""
    return CardioPredictor().predict(audio_path).model_dump()