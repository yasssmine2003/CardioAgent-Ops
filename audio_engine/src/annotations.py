"""Lecture des annotations de référence PhysioNet 2016 (S1, systole, S2, diastole).

Chaque fichier <nom>_StateAns.mat contient une liste de lignes (instant, état) :
l'instant (en échantillons à 2000 Hz) où commence chaque état. Un état dure
jusqu'au début du suivant. Ces annotations ont été produites par l'algorithme de
Springer puis corrigées à la main.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src import config

STATE_NAMES = {"s1": "S1", "s2": "S2", "sys": "systole", "dia": "diastole"}


@dataclass
class Reference:
    """Annotation de référence d'un enregistrement."""
    states: list[tuple[str, float, float]]      # (état, début s, fin s)

    def intervals(self, state: str) -> np.ndarray:
        return np.array([(s, e) for name, s, e in self.states if name == state]).reshape(-1, 2)

    @property
    def s1_starts(self) -> np.ndarray:
        return self.intervals("S1")[:, 0]

    @property
    def heart_rate_bpm(self) -> float:
        """Rythme de référence = 60 / durée médiane entre deux S1 successifs."""
        gaps = np.diff(self.s1_starts)
        return float(60 / np.median(gaps)) if len(gaps) else float("nan")


def normalize_state(label: str) -> str:
    label = label.strip().lower()
    for key, name in STATE_NAMES.items():
        if key in label:
            return name
    return "autre"                                # bruit, état inconnu...


def load_reference(path: Path, sr: int = config.TARGET_SR) -> Reference | None:
    """Lit un fichier _StateAns.mat. Retourne None s'il est vide (enregistrement non annoté)."""
    content = loadmat(path)
    keys = [k for k in content if not k.startswith("__")]
    if not keys:
        return None
    table = np.asarray(content[keys[0]], dtype=object)
    if table.size == 0:
        return None
    table = table.reshape(-1, 2)
    starts = [float(np.squeeze(row[0])) / sr for row in table]
    labels = [normalize_state(str(np.squeeze(row[1]))) for row in table]
    states = [(label, start, end) for label, start, end in zip(labels, starts, starts[1:] + [starts[-1]])
              if end > start]
    return Reference(states) if any(s[0] == "S1" for s in states) else None