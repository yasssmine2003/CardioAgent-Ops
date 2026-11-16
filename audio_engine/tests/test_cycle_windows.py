"""Tests du découpage en fenêtres alignées sur les cycles."""
from src.cycle_windows import cycle_windows, runs_of_valid_cycles
from src.segmentation import Cycle


def make_cycles(durations, invalid=()):
    cycles, t = [], 0.0
    for k, d in enumerate(durations):
        cycles.append(Cycle(t, t + d, d, k not in invalid, "ok" if k not in invalid else "duree"))
        t += d
    return cycles


def test_windows_contain_only_complete_cycles_and_last_at_most_5s():
    windows = cycle_windows(make_cycles([0.8] * 12))     # 9,6 s de cycles réguliers
    assert [w.n_cycles for w in windows] == [6, 6]        # 6 x 0,8 = 4,8 s <= 5 s
    assert all(w.end_s - w.start_s <= 5.0 for w in windows)
    assert windows[1].start_s == windows[0].end_s         # pas de chevauchement


def test_invalid_cycle_splits_the_run():
    cycles = make_cycles([0.8] * 6, invalid={2})
    assert [len(r) for r in runs_of_valid_cycles(cycles)] == [2, 3]
    assert [w.n_cycles for w in cycle_windows(cycles)] == [2, 3]


def test_single_cycle_is_not_a_window():
    assert cycle_windows(make_cycles([0.8, 0.8, 0.8], invalid={1})) == []
