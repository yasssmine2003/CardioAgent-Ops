"""Prétraitement audio du Service 3 : chargement, rééchantillonnage, normalisation, segmentation.

Les mêmes fonctions servent à l'entraînement ET à l'inférence. C'est la garantie
que le modèle verra en production exactement le même type de données qu'à l'entraînement.

Pipeline :
    fichier .wav -> load_audio -> validate_signal -> resample_audio (2000 Hz)
                 -> normalize_audio -> segment_audio (5 s) -> segments
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.config import (MIN_DURATION_S, SEGMENT_DURATION_S, SEGMENT_HOP_S,
                        SILENCE_STD_THRESHOLD, TARGET_SR)

logger = logging.getLogger(__name__)


class AudioError(ValueError):
    """Audio inutilisable : fichier illisible, vide, silencieux, trop court, NaN..."""


@dataclass(frozen=True)
class PreprocessedAudio:
    """Résultat du prétraitement d'un enregistrement."""
    segments: np.ndarray   # forme (n_segments, SEGMENT_DURATION_S * TARGET_SR), float32
    original_sr: int       # fréquence d'origine (Hz)
    duration_s: float      # durée de l'enregistrement d'origine (s)
    padded: bool           # True si l'audio a été complété jusqu'à SEGMENT_DURATION_S


# ============================ 1. CHARGEMENT ============================

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Charge un fichier audio en mono, à sa fréquence d'origine (aucun rééchantillonnage).

    Returns:
        (signal 1D float32, fréquence d'origine en Hz)
    Raises:
        AudioError : fichier introuvable, illisible (mauvais format, corrompu) ou vide.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Fichier introuvable : {path}")
    try:
        # sr=None : on garde la fréquence d'origine ; mono=True : stéréo -> moyenne des canaux
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception as exc:
        raise AudioError(f"Fichier illisible ({path.name}) : {exc}") from exc
    if y.size == 0:
        raise AudioError(f"Fichier vide : {path.name}")
    return y.astype(np.float32), int(sr)


def validate_signal(y: np.ndarray) -> np.ndarray:
    """Vérifie un signal et le met au format standard : 1D (mono), float32, sans NaN/Inf.

    Accepte aussi un signal stéréo 2D (l'axe le plus court est celui des canaux).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        channel_axis = 0 if y.shape[0] <= y.shape[1] else 1
        y = y.mean(axis=channel_axis)
    if y.ndim != 1:
        raise AudioError(f"Forme de signal non supportée : {y.shape}")
    if y.size == 0:
        raise AudioError("Signal vide")
    if not np.all(np.isfinite(y)):
        raise AudioError("Signal contenant des valeurs NaN ou infinies")
    return y


# ========================= 2. RÉÉCHANTILLONNAGE =========================

def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Ramène le signal à target_sr Hz (2000 Hz par défaut, imposé par le cahier des charges).

    librosa applique un filtre anti-repliement avant de réduire le nombre d'échantillons.
    """
    if orig_sr <= 0:
        raise AudioError(f"Fréquence d'échantillonnage invalide : {orig_sr}")
    if orig_sr == target_sr:
        return y  # PhysioNet : déjà à 2000 Hz, rien à faire
    if orig_sr < target_sr:
        logger.warning("Sur-échantillonnage %d -> %d Hz : aucune information ajoutée", orig_sr, target_sr)
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


# =========================== 3. NORMALISATION ===========================

def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalisation d'amplitude par z-score : moyenne 0, écart-type 1.

    1. On retire la moyenne (décalage continu propre à chaque micro).
    2. On divise par l'écart-type : un même cœur enregistré fort ou faible
       donne le même signal normalisé, quelle que soit la source.

    Raises:
        AudioError si le signal est silencieux (écart-type quasi nul).
    """
    y = y - y.mean()
    std = float(y.std())
    if std < SILENCE_STD_THRESHOLD:
        raise AudioError(f"Signal silencieux (écart-type {std:.2e})")
    return (y / std).astype(np.float32)


# ============================ 4. SEGMENTATION ============================

def pad_by_repetition(y: np.ndarray, target_length: int) -> np.ndarray:
    """Complète un signal court en le répétant jusqu'à target_length échantillons.

    On répète le son (au lieu d'ajouter du silence) pour que le segment complété
    contienne des battements du début à la fin, comme un segment normal.
    """
    return np.resize(y, target_length)  # np.resize répète le tableau en boucle


def segment_audio(y: np.ndarray, sr: int = TARGET_SR,
                  segment_s: float = SEGMENT_DURATION_S, hop_s: float = SEGMENT_HOP_S,
                  min_s: float = MIN_DURATION_S) -> np.ndarray:
    """Découpe le signal en segments de durée fixe.

    - durée >= segment_s           : fenêtres successives ; la fin trop courte est ignorée
    - min_s <= durée < segment_s   : un seul segment, complété par répétition
    - durée < min_s                : AudioError (trop court pour être fiable)

    Returns:
        tableau (n_segments, segment_s * sr)
    """
    seg_len = int(round(segment_s * sr))
    hop_len = int(round(hop_s * sr))
    min_len = int(round(min_s * sr))
    n = len(y)

    if n < min_len:
        raise AudioError(f"Audio trop court : {n / sr:.2f} s < {min_s} s")
    if n < seg_len:
        return pad_by_repetition(y, seg_len)[np.newaxis, :]

    starts = range(0, n - seg_len + 1, hop_len)
    return np.stack([y[s:s + seg_len] for s in starts]).astype(np.float32)


# ============================ 5. PIPELINE ============================

def preprocess_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Pipeline complet sur un signal déjà chargé (utilisé aussi à l'inférence)."""
    y = validate_signal(y)
    y = resample_audio(y, sr)
    y = normalize_audio(y)
    return segment_audio(y)


def preprocess_file(path: str | Path) -> PreprocessedAudio:
    """Pipeline complet sur un fichier : chargement -> ... -> segments."""
    y, sr = load_audio(path)
    duration_s = len(y) / sr
    segments = preprocess_signal(y, sr)
    return PreprocessedAudio(segments=segments, original_sr=sr, duration_s=duration_s,
                             padded=duration_s < SEGMENT_DURATION_S)