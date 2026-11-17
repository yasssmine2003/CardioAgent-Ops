"""Segmentation d'un enregistrement cardiaque en cycles (approche 2 : autocorrélation).

Idée : mesurer d'abord le rythme propre au patient, puis chercher un battement par cycle.

    signal (2000 Hz, z-score)
      -> ⓪ repérage des coupures                    (passages plats : appareil, stéthoscope décollé)
      -> ① filtre passe-bande 25-400 Hz            (garder les bruits du cœur)
      -> ② enveloppe homomorphique à 50 Hz          (une bosse par bruit)
      -> ③ autocorrélation sur fenêtres glissantes  (durée du cycle, propre au patient,
                                                    jamais une borne de la zone de recherche)
      -> ④ détection des battements                 (un pic dominant par cycle,
                                                    puis correction des erreurs de demi-cycle)
      -> ⑤ découpage en cycles                      (d'un battement au suivant)
      -> ⑥ contrôle qualité                         (cycles de durée anormale signalés)

Limite assumée : le repère de chaque cycle est le bruit le plus fort (S1 ou S2),
sans distinguer S1 de S2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, resample_poly

from src import config


@dataclass
class Cycle:
    """Un cycle cardiaque détecté."""
    start_s: float
    end_s: float
    expected_s: float      # durée attendue d'après le rythme local
    valid: bool            # cycle utilisable ?
    reason: str = "ok"     # "ok", "duree" (durée anormale) ou "coupure" (chevauche une coupure)

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class CycleSegmentation:
    """Résultat de la segmentation d'un enregistrement."""
    duration_s: float                    # durée de l'enregistrement
    cycle_duration_s: float              # durée typique d'un cycle (médiane des fenêtres)
    heart_rate_bpm: float                # rythme cardiaque = 60 / durée du cycle
    periodicity: float                   # 0 à 1 : à quel point le signal est rythmique
    beat_times_s: np.ndarray             # instants des battements détectés
    cycles: list[Cycle] = field(default_factory=list)
    silences: list[tuple[float, float]] = field(default_factory=list)   # coupures (début, fin) en s

    @property
    def silence_s(self) -> float:
        """Durée totale des coupures."""
        return round(sum(end - start for start, end in self.silences), 3)

    @property
    def valid_cycles(self) -> list[Cycle]:
        return [c for c in self.cycles if c.valid]

    @property
    def irregularity(self) -> float:
        """Variabilité des cycles valides (écart-type / moyenne) : 0 = parfaitement régulier."""
        durations = np.array([c.duration_s for c in self.valid_cycles])
        return round(float(durations.std() / durations.mean()), 3) if len(durations) >= 2 else float("nan")

    @property
    def reliable(self) -> bool:
        """La segmentation est-elle exploitable ?"""
        return (self.periodicity >= config.SEG_MIN_PERIODICITY
                and len(self.valid_cycles) >= config.SEG_MIN_VALID_CYCLES)


# ============================ ⓪ Coupures ============================

def find_silences(y: np.ndarray, sr: int = config.TARGET_SR) -> list[tuple[float, float]]:
    """⓪ Repère les coupures : passages d'au moins 0,5 s où le signal ne VARIE plus.

    On découpe le signal en trames de 50 ms et on mesure la variation de chacune
    (écart-type). Une trame est « plate » si sa variation est inférieure à 2 % de la
    variation médiane de l'enregistrement : un seuil relatif, valable pour tout appareil.

    Pourquoi la variation et pas l'énergie ? Après la normalisation z-score, une coupure
    n'est pas à zéro : elle devient une ligne plate DÉCALÉE (on a retiré la moyenne du
    signal). Son énergie n'est donc pas nulle, mais sa variation, elle, l'est.
    (Une diastole est calme, mais elle varie toujours un peu.)
    """
    frame = int(config.SEG_SILENCE_FRAME_S * sr)
    n_frames = len(y) // frame
    if n_frames == 0:
        return []
    variation = y[:n_frames * frame].reshape(n_frames, frame).std(axis=1)
    flat = variation < config.SEG_SILENCE_RATIO * np.median(variation)
    silences, start = [], None
    for i, is_flat in enumerate(np.append(flat, False)):         # False final : ferme la dernière coupure
        if is_flat and start is None:
            start = i
        elif not is_flat and start is not None:
            if (i - start) * config.SEG_SILENCE_FRAME_S >= config.SEG_MIN_SILENCE_S:
                silences.append((round(start * config.SEG_SILENCE_FRAME_S, 3),
                                 round(i * config.SEG_SILENCE_FRAME_S, 3)))
            start = None
    return silences


def in_silence(t: float, silences: list[tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in silences)


def neutralize_silences(envelope: np.ndarray, silences: list[tuple[float, float]],
                        env_sr: int = config.SEG_ENVELOPE_SR) -> np.ndarray:
    """Remplace l'enveloppe des coupures par sa valeur médiane : elles ne créent
    ni faux battements ni faux rythme dans l'autocorrélation."""
    envelope = envelope.copy()
    for start, end in silences:
        envelope[int(start * env_sr):int(end * env_sr) + 1] = np.median(envelope)
    return envelope


# ============================ ① ② Enveloppe ============================

def bandpass(y: np.ndarray, sr: int = config.TARGET_SR) -> np.ndarray:
    """① Garde la bande des bruits cardiaques (25-400 Hz)."""
    b, a = butter(4, config.SEG_BAND_HZ, btype="band", fs=sr)
    return filtfilt(b, a, y)


def homomorphic_envelope(y: np.ndarray, sr: int = config.TARGET_SR,
                         env_sr: int = config.SEG_ENVELOPE_SR) -> np.ndarray:
    """② Enveloppe homomorphique normalisée (moyenne 0, écart-type 1), à env_sr Hz.

    exp(passe-bas(log |Hilbert|)) : grâce au logarithme, un claquement très fort
    déforme beaucoup moins la courbe qu'avec un lissage direct.
    """
    amplitude = np.abs(hilbert(bandpass(y, sr))) + 1e-8
    b, a = butter(1, config.SEG_ENVELOPE_CUTOFF_HZ, btype="low", fs=sr)
    envelope = np.exp(filtfilt(b, a, np.log(amplitude)))
    envelope = resample_poly(envelope, env_sr, sr)            # 2000 Hz -> 50 Hz
    return (envelope - envelope.mean()) / (envelope.std() + 1e-8)


# ============================ ③ Rythme du patient ============================

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Autocorrélation normalisée (1 au décalage 0), calculée rapidement par FFT."""
    x = x - x.mean()
    spectrum = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:len(x)]
    return ac / (ac[0] + 1e-12)


def estimate_cycle_duration(envelope: np.ndarray,
                            env_sr: int = config.SEG_ENVELOPE_SR) -> tuple[float, float]:
    """Durée du cycle = plus haut pic d'autocorrélation entre 40 et 150 bpm.

    On ne retient que de VRAIS pics (maximums locaux), jamais une borne de la zone de
    recherche : un résultat pile à 40 ou 150 bpm était un artefact de la borne.
    (Les erreurs de demi-cycle sont corrigées ensuite, sur les battements : voir
    half_cycle_detected.)

    Retourne (durée en s, périodicité entre 0 et 1 = hauteur du pic retenu).
    """
    ac = autocorrelation(envelope)
    lo = int(60 / config.SEG_BPM_RANGE[1] * env_sr)
    hi = min(int(60 / config.SEG_BPM_RANGE[0] * env_sr), len(ac) - 2)
    if hi <= lo:
        return float("nan"), 0.0
    peaks, _ = find_peaks(ac[:hi + 2])
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]                 # vrais pics, dans la zone
    if len(peaks) == 0:
        return float("nan"), 0.0
    lag = int(peaks[np.argmax(ac[peaks])])
    return lag / env_sr, float(max(ac[lag], 0.0))


def local_cycle_durations(envelope: np.ndarray, env_sr: int = config.SEG_ENVELOPE_SR
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """③ Durée du cycle sur des fenêtres glissantes de 5 s : suit un rythme qui change.

    Retourne (centres des fenêtres en s, durées en s, périodicités).
    """
    win, hop = int(config.SEG_WINDOW_S * env_sr), int(config.SEG_HOP_S * env_sr)
    if len(envelope) <= win:                                  # enregistrement court : une seule fenêtre
        duration, strength = estimate_cycle_duration(envelope, env_sr)
        return np.array([len(envelope) / env_sr / 2]), np.array([duration]), np.array([strength])
    centers, durations, strengths = [], [], []
    for start in range(0, len(envelope) - win + 1, hop):
        duration, strength = estimate_cycle_duration(envelope[start:start + win], env_sr)
        centers.append((start + win / 2) / env_sr)
        durations.append(duration)
        strengths.append(strength)
    return np.array(centers), np.array(durations), np.array(strengths)


# ============================ ④ ⑤ ⑥ Battements et cycles ============================

def detect_beats(envelope: np.ndarray, centers: np.ndarray, durations: np.ndarray,
                 env_sr: int = config.SEG_ENVELOPE_SR) -> np.ndarray:
    """④ Un pic dominant par cycle, avec une distance minimale qui suit le rythme local.

    1. On repère tous les pics candidats (bosses de l'enveloppe).
    2. On les examine du plus fort au plus faible.
    3. On garde un pic s'il n'y a pas déjà un pic gardé à moins de 0,6 x le cycle local :
       le plus fort bruit de chaque cycle est retenu, le plus faible (souvent l'autre bruit) est écarté.
    """
    candidates, props = find_peaks(envelope, distance=max(1, int(0.15 * env_sr)), prominence=0.5)
    times = candidates / env_sr
    kept: list[float] = []
    for i in np.argsort(props["prominences"])[::-1]:               # du plus fort au plus faible
        min_gap = config.SEG_MIN_PEAK_SPACING * float(np.interp(times[i], centers, durations))
        if all(abs(times[i] - k) >= min_gap for k in kept):
            kept.append(float(times[i]))
    return np.sort(np.array(kept))


def alternation(values: np.ndarray) -> float:
    """Mesure si une suite alterne « grand, petit, grand, petit ».

    Compare l'écart entre voisins (1 et 2, 2 et 3...) à l'écart entre éléments
    séparés d'un rang (1 et 3, 2 et 4...). Une suite qui alterne a de grands écarts
    entre voisins mais de petits écarts un rang sur deux : le rapport est grand.
    Les écarts de moins de 15 % de la valeur moyenne sont ignorés (imprécision de mesure).
    """
    if len(values) < 5:
        return 0.0
    neighbours = np.mean(np.abs(np.diff(values)))
    if neighbours < config.SEG_ALTERNATION_MIN * np.mean(np.abs(values)):
        return 0.0                         # écarts trop petits (< 15 %) : simple imprécision, pas une alternance
    every_other = np.mean(np.abs(values[2:] - values[:-2]))
    return float(neighbours / (every_other + 1e-8))


def beat_descriptors(y_band: np.ndarray, beat_times_s: np.ndarray,
                     sr: int = config.TARGET_SR) -> tuple[np.ndarray, np.ndarray]:
    """Pour chaque battement : son énergie et sa « hauteur » (centroïde spectral, en Hz).

    Mesurées sur le signal filtré à 2000 Hz, dans une fenêtre de +/- 40 ms autour du pic.
    (Mesurer la hauteur de l'enveloppe à 50 Hz serait trop imprécis.)
    """
    half = int(0.04 * sr)
    energies, centroids = [], []
    for t in beat_times_s:
        i = int(t * sr)
        window = y_band[max(i - half, 0):i + half]
        spectrum = np.abs(np.fft.rfft(window)) ** 2              # puissance : la fréquence dominante pèse plus que le bruit
        freqs = np.fft.rfftfreq(len(window), 1 / sr)
        energies.append(np.sqrt(np.mean(window ** 2)))
        centroids.append(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12))
    return np.array(energies), np.array(centroids)


def half_cycle_detected(y_band: np.ndarray, beat_times_s: np.ndarray,
                        sr: int = config.TARGET_SR) -> bool:
    """Erreur de demi-cycle : les « battements » détectés sont en réalité S1 et S2 en alternance.

    Trois signes possibles, qui alternent d'un battement au suivant :
    - l'énergie   : S1 et S2 n'ont pas la même force ;
    - la hauteur  : S2 est en général plus aigu que S1 ;
    - les écarts  : systole (S1 -> S2) plus courte que diastole (S2 -> S1).
    Le vrai cycle est alors deux fois plus long.
    """
    if len(beat_times_s) < 5:
        return False
    energies, centroids = beat_descriptors(y_band, beat_times_s, sr)
    gaps = np.diff(beat_times_s)
    return max(alternation(energies), alternation(centroids), alternation(gaps)) >= config.SEG_ALTERNATION_RATIO


def build_cycles(beat_times_s: np.ndarray, centers: np.ndarray, durations: np.ndarray,
                 total_s: float, silences: list[tuple[float, float]]) -> list[Cycle]:
    """⑤ Un cycle va d'un battement au suivant (en commençant un peu avant le pic).
    ⑥ Contrôle qualité : il est rejeté s'il chevauche une coupure ("coupure"),
       ou si sa durée s'écarte de plus de 30 % de la durée de cycle locale ("duree")."""
    cycles = []
    for t0, t1 in zip(beat_times_s[:-1], beat_times_s[1:]):
        expected = float(np.interp(t0, centers, durations))     # rythme local au début du cycle
        start = max(t0 - config.SEG_PRE_PEAK * expected, 0.0)
        end = min(t1 - config.SEG_PRE_PEAK * expected, total_s)
        if any(s0 < end and s1 > start for s0, s1 in silences):
            reason = "coupure"
        elif abs((end - start) - expected) > config.SEG_TOLERANCE * expected:
            reason = "duree"
        else:
            reason = "ok"
        cycles.append(Cycle(round(start, 3), round(end, 3), round(expected, 3), reason == "ok", reason))
    return cycles


def segment_cycles(y: np.ndarray, sr: int = config.TARGET_SR) -> CycleSegmentation:
    """Pipeline complet ⓪→⑥ sur un signal prétraité (2000 Hz, z-score)."""
    total_s = len(y) / sr
    silences = find_silences(y, sr)
    envelope = neutralize_silences(homomorphic_envelope(y, sr), silences)
    centers, durations, strengths = local_cycle_durations(envelope)
    ok = np.isfinite(durations)
    if not ok.any():                                          # signal trop court pour mesurer un rythme
        return CycleSegmentation(round(total_s, 3), float("nan"), float("nan"), 0.0, np.array([]),
                                 silences=silences)

    centers, durations, strengths = centers[ok], durations[ok], strengths[ok]
    beats = np.array([t for t in detect_beats(envelope, centers, durations) if not in_silence(t, silences)])
    # Correction de l'erreur de demi-cycle : on a pris S1 ET S2 pour des battements
    if half_cycle_detected(bandpass(y, sr), beats, sr) and 2 * np.median(durations) <= 60 / config.SEG_BPM_RANGE[0]:
        durations = 2 * durations
        beats = np.array([t for t in detect_beats(envelope, centers, durations) if not in_silence(t, silences)])
    cycle_s = float(np.median(durations))
    return CycleSegmentation(
        duration_s=round(total_s, 3),
        cycle_duration_s=round(cycle_s, 3),
        heart_rate_bpm=round(60 / cycle_s, 1),
        periodicity=round(float(np.median(strengths)), 3),
        beat_times_s=np.round(beats, 3),
        cycles=build_cycles(beats, centers, durations, total_s, silences),
        silences=silences,
    )