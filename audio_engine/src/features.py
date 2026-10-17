"""Extraction des caractéristiques audio du Service 3 : MFCC, ZCR, spectrogrammes.

Entrée de toutes les fonctions : UN segment prétraité (1D, 2000 Hz, 5 s = 10 000 valeurs).

Deux représentations sont produites pour les modèles :
- un vecteur 1D de 28 valeurs (MFCC + ZCR résumés)  -> Decision Tree, SVM
- un Mel-spectrogramme 2D (32 x 157), en dB          -> CNN, ResNet18, Transformer-CNN

Les mêmes fonctions servent à l'entraînement ET à l'inférence.
"""
from __future__ import annotations

import librosa
import numpy as np

from src.config import (FMAX, HOP_LENGTH, N_FFT, N_MELS, N_MFCC,
                        SEGMENT_DURATION_S, TARGET_SR)

# Nombre d'instants (colonnes) par segment : 1 + 10 000 // 64 = 157
N_FRAMES: int = 1 + int(SEGMENT_DURATION_S * TARGET_SR) // HOP_LENGTH
MEL_SHAPE: tuple[int, int] = (N_MELS, N_FRAMES)

FEATURE_NAMES: list[str] = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + ["zcr_mean", "zcr_std"]
)

# ==================== Représentations temps-fréquence ====================

def compute_spectrogram(segment: np.ndarray) -> np.ndarray:
    """Spectrogramme classique (STFT) en dB.

    Returns:
        (1 + N_FFT // 2, N_FRAMES) = (129, 157) : fréquences linéaires x instants
    """
    magnitude = np.abs(librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return librosa.amplitude_to_db(magnitude, ref=1.0).astype(np.float32)


def compute_mel_spectrogram(segment: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Mel-spectrogramme en dB : l'« image » d'entrée des modèles de deep learning.

    Returns:
        (N_MELS, N_FRAMES) = (32, 157) : bandes Mel x instants
    """
    mel_power = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=N_FFT,
                                               hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
    return librosa.power_to_db(mel_power, ref=1.0, top_db=80.0).astype(np.float32)


# ============================ MFCC et ZCR ============================

def compute_mfcc(segment: np.ndarray, sr: int = TARGET_SR,
                 mel_db: np.ndarray | None = None) -> np.ndarray:
    """MFCC : description compacte de la forme du spectre, à chaque instant.

    Si le Mel-spectrogramme est déjà calculé, on le réutilise (pas de double calcul).

    Returns:
        (N_MFCC, N_FRAMES) = (13, 157)
    """
    if mel_db is None:
        mel_db = compute_mel_spectrogram(segment, sr)
    return librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC).astype(np.float32)


def compute_zcr(segment: np.ndarray) -> np.ndarray:
    """Zero-Crossing Rate : proportion de passages par zéro, à chaque instant (entre 0 et 1).

    Returns:
        (N_FRAMES,) = (157,)
    """
    zcr = librosa.feature.zero_crossing_rate(segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    return zcr[0].astype(np.float32)


# ========================= Caractéristiques finales =========================

def summarize_1d(mfcc: np.ndarray, zcr: np.ndarray) -> np.ndarray:
    """Résume un segment en un vecteur 1D : moyenne et écart-type de chaque coefficient.

    Returns:
        (28,) = 13 moyennes MFCC + 13 écarts-types MFCC + moyenne ZCR + écart-type ZCR
    """
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                           [zcr.mean(), zcr.std()]]).astype(np.float32)


def extract_features(segment: np.ndarray, sr: int = TARGET_SR) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les deux représentations d'un segment en une seule passe.

    Returns:
        (vecteur 1D (28,), Mel-spectrogramme (32, 157))
    """
    mel_db = compute_mel_spectrogram(segment, sr)
    mfcc = compute_mfcc(segment, sr, mel_db=mel_db)
    zcr = compute_zcr(segment)
    return summarize_1d(mfcc, zcr), mel_db