"""Tests de la lecture des annotations PhysioNet et de la comparaison avec la référence."""
import numpy as np
import pytest
from scipy.io import savemat

from src.annotations import Reference, load_reference, normalize_state
from src.segmentation import CycleSegmentation
from src.segmentation_eval import compare


def reference(period=1.0, n=10, s2_offset=0.35):
    """Référence : S1 (80 ms) puis systole, S2 (80 ms) puis diastole, toutes les `period` s."""
    states = []
    for k in range(n):
        t = k * period
        states += [("S1", t, t + 0.08), ("systole", t + 0.08, t + s2_offset),
                   ("S2", t + s2_offset, t + s2_offset + 0.08), ("diastole", t + s2_offset + 0.08, t + period)]
    return Reference(states)


def segmentation(beats, bpm=60.0):
    return CycleSegmentation(10.0, 60 / bpm, bpm, 0.8, np.array(beats))


def test_state_names_are_normalized():
    assert [normalize_state(s) for s in ["S1", "systole", "S2", "Diastole", "(N"]] == \
           ["S1", "systole", "S2", "diastole", "autre"]


def test_load_reference_from_mat(tmp_path):
    """Lecture d'un fichier au format (instant en échantillons, état)."""
    table = np.empty((4, 2), dtype=object)
    for i, (idx, state) in enumerate([(0, "S1"), (160, "systole"), (700, "S2"), (860, "diastole")]):
        table[i, 0], table[i, 1] = np.array([[idx]]), np.array([state])
    savemat(tmp_path / "x_StateAns.mat", {"state_ans": table})
    ref = load_reference(tmp_path / "x_StateAns.mat")
    assert ref.states[0] == ("S1", 0.0, 0.08)


def test_perfect_segmentation():
    """Un repère sur chaque S1 : tout est juste."""
    m = compare(segmentation([k + 0.04 for k in range(10)]), reference())
    assert m["beat_precision"] == 1.0 and m["landmark_on_s1"] == 1.0 and m["cycles_ok"] == 1.0
    assert m["hr_error_pct"] == pytest.approx(0.0)


def test_half_cycle_error_is_visible():
    """Repères sur S1 ET S2 : chaque cycle est coupé en deux, et le rythme est doublé."""
    beats = sorted([k + 0.04 for k in range(10)] + [k + 0.39 for k in range(10)])
    m = compare(segmentation(beats, bpm=120.0), reference())
    assert m["cycles_split"] == 1.0
    assert m["hr_error_pct"] == pytest.approx(100.0)


def test_missed_beats_are_counted():
    """Un repère sur deux cycles seulement : la moitié des cycles est manquée."""
    m = compare(segmentation([k + 0.04 for k in range(0, 10, 2)], bpm=30.0), reference())
    assert m["cycles_missed"] == pytest.approx(0.5, abs=0.12)