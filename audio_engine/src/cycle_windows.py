"""Fenêtres de 5 s alignées sur les cycles cardiaques (option A).

Idée : au lieu de couper l'audio toutes les 5 s (au hasard du rythme), chaque fenêtre
commence au début d'un battement et ne contient que des cycles complets et valides.
La fenêtre est ensuite complétée par répétition jusqu'à 5 s -> même entrée (32 x 157)
pour les 5 modèles, rien d'autre ne change.

Si la segmentation n'est pas fiable, on revient aux fenêtres fixes (repli).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.preprocessing import pad_by_repetition, segment_audio
from src.segmentation import Cycle, CycleSegmentation, segment_cycles

MIN_CYCLES_PER_WINDOW = 2   # [C] au moins 2 cycles complets par fenêtre
CONTIGUITY_TOL_S = 0.02     # deux cycles sont « consécutifs » si l'écart est < 20 ms


@dataclass
class Window:
    start_s: float
    end_s: float
    n_cycles: int           # 0 = fenêtre fixe (repli)
    mode: str               # "cycles" ou "fixed"


def runs_of_valid_cycles(cycles: list[Cycle]) -> list[list[Cycle]]:
    """Regroupe les cycles valides qui se suivent sans interruption."""
    runs, current = [], []
    for c in cycles:
        if c.valid and (not current or abs(c.start_s - current[-1].end_s) < CONTIGUITY_TOL_S):
            current.append(c)
        else:
            if current:
                runs.append(current)
            current = [c] if c.valid else []
    if current:
        runs.append(current)
    return runs


def cycle_windows(cycles: list[Cycle], max_s: float = config.SEGMENT_DURATION_S) -> list[Window]:
    """Remplit chaque fenêtre avec le plus de cycles complets possible (≤ 5 s), sans chevauchement."""
    windows = []
    for run in runs_of_valid_cycles(cycles):
        i = 0
        while i < len(run):
            j = i
            while j + 1 < len(run) and run[j + 1].end_s - run[i].start_s <= max_s:
                j += 1
            n = j - i + 1
            if n >= MIN_CYCLES_PER_WINDOW and run[j].end_s - run[i].start_s <= max_s:
                windows.append(Window(float(max(0.0, run[i].start_s)), float(run[j].end_s), n, "cycles"))
            i = j + 1
    return windows


def build_windows(y: np.ndarray, sr: int = config.TARGET_SR
                  ) -> tuple[np.ndarray, list[Window], CycleSegmentation | None]:
    """Signal prétraité (2000 Hz, z-score) -> (fenêtres (n, 10000), description, segmentation).

    Lève AudioError si l'audio fait moins de 2 s (même règle qu'avant).
    """
    seg_len = int(round(config.SEGMENT_DURATION_S * sr))
    fixed = segment_audio(y, sr)                       # contrôle « < 2 s » + repli
    seg = segment_cycles(y, sr)
    windows = cycle_windows(seg.cycles) if seg.reliable else []

    if not windows:                                    # repli : fenêtres fixes
        hop = config.SEGMENT_HOP_S
        windows = [Window(k * hop, min(k * hop + config.SEGMENT_DURATION_S, len(y) / sr), 0, "fixed")
                   for k in range(len(fixed))]
        return fixed, windows, seg

    segments = np.stack([
        pad_by_repetition(y[int(round(w.start_s * sr)):int(round(w.end_s * sr))], seg_len)
        for w in windows
    ]).astype(np.float32)
    return segments, windows, seg
