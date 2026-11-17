"""Comparaison d'une segmentation (approche 2) avec l'annotation de référence PhysioNet.

Trois mesures par enregistrement :
- erreur de rythme       : écart entre notre rythme et celui de la référence (en %) ;
- précision des repères  : part de nos battements qui tombent sur un vrai bruit S1 ou S2 ;
- cycles bien découpés   : part des vrais cycles qui contiennent EXACTEMENT un de nos repères
                           (0 = cycle manqué, 2 ou plus = cycle coupé en deux).
"""
from __future__ import annotations

import numpy as np

from src import config
from src.annotations import Reference
from src.segmentation import CycleSegmentation


def inside(t: float, intervals: np.ndarray, margin: float = config.EVAL_MARGIN_S) -> bool:
    return bool(np.any((intervals[:, 0] - margin <= t) & (t <= intervals[:, 1] + margin)))


def compare(seg: CycleSegmentation, ref: Reference) -> dict:
    s1, s2 = ref.intervals("S1"), ref.intervals("S2")
    first, last = ref.states[0][1], ref.states[-1][2]
    beats = [t for t in seg.beat_times_s if first <= t <= last]       # zone annotée seulement

    on_s1 = sum(inside(t, s1) for t in beats)
    on_s2 = sum(inside(t, s2) and not inside(t, s1) for t in beats)
    starts = ref.s1_starts
    beats_arr = np.array(beats)
    per_cycle = [int(((beats_arr >= a - config.EVAL_MARGIN_S) & (beats_arr < b - config.EVAL_MARGIN_S)).sum())
                 for a, b in zip(starts[:-1], starts[1:])]

    hr_ref = ref.heart_rate_bpm
    return {
        "hr_ref_bpm": round(hr_ref, 1),
        "hr_est_bpm": seg.heart_rate_bpm,
        "hr_error_pct": round(abs(seg.heart_rate_bpm - hr_ref) / hr_ref * 100, 1),
        "n_beats": len(beats),
        "beat_precision": round((on_s1 + on_s2) / len(beats), 3) if beats else float("nan"),
        "landmark_on_s1": round(on_s1 / (on_s1 + on_s2), 3) if (on_s1 + on_s2) else float("nan"),
        "n_ref_cycles": len(per_cycle),
        "cycles_ok": round(np.mean([n == 1 for n in per_cycle]), 3) if per_cycle else float("nan"),
        "cycles_missed": round(np.mean([n == 0 for n in per_cycle]), 3) if per_cycle else float("nan"),
        "cycles_split": round(np.mean([n >= 2 for n in per_cycle]), 3) if per_cycle else float("nan"),
    }