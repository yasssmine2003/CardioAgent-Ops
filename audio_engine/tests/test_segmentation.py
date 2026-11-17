"""Tests de la segmentation par cycles (approche 2), sur des battements synthétiques."""
import numpy as np
import pytest

from src.config import TARGET_SR
from src.segmentation import autocorrelation, homomorphic_envelope, segment_cycles


def heartbeat(periods_s, duration_s, noise=0.05, seed=0):
    """Son cardiaque synthétique : un S1 (50 Hz) et un S2 plus faible (70 Hz) par cycle.

    periods_s : durée du cycle, constante (float) ou variable (fonction du temps).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * TARGET_SR)) / TARGET_SR
    y = noise * rng.standard_normal(len(t))
    period = periods_s if callable(periods_s) else (lambda _: periods_s)
    beat = 0.2
    while beat < duration_s - 0.2:
        p = period(beat)
        for offset, freq, amp in [(0.0, 50, 1.0), (0.35 * p, 70, 0.6)]:
            m = (t >= beat + offset) & (t < beat + offset + 0.08)
            y[m] += amp * np.sin(2 * np.pi * freq * t[m]) * np.hanning(m.sum())
        beat += p
    return ((y - y.mean()) / y.std()).astype(np.float32)


def test_autocorrelation_is_one_at_zero_lag():
    ac = autocorrelation(np.sin(np.linspace(0, 20, 500)))
    assert ac[0] == pytest.approx(1.0)


def test_envelope_is_at_50_hz():
    """2000 points par seconde -> 50 points par seconde."""
    env = homomorphic_envelope(heartbeat(0.8, 10))
    assert len(env) == pytest.approx(10 * 50, abs=2)


@pytest.mark.parametrize("period, bpm", [(0.6, 100), (0.8, 75), (1.2, 50)])
def test_heart_rate_is_measured(period, bpm):
    """Le rythme du patient est retrouvé, quel qu'il soit."""
    seg = segment_cycles(heartbeat(period, 20))
    assert seg.heart_rate_bpm == pytest.approx(bpm, abs=4)
    assert seg.reliable
    assert len(seg.valid_cycles) >= 0.8 * (20 / period - 2)


def test_cycles_follow_a_changing_rhythm():
    """Le cœur accélère de 60 à 100 bpm : les cycles restent valides."""
    seg = segment_cycles(heartbeat(lambda t: 1.0 if t < 15 else 0.6, 30))
    assert len(seg.valid_cycles) / len(seg.cycles) >= 0.8
    first, last = seg.valid_cycles[0].duration_s, seg.valid_cycles[-1].duration_s
    assert first > last                      # les derniers cycles sont plus courts


def test_noise_only_is_not_reliable():
    """Du bruit sans battements ne doit pas être déclaré fiable."""
    seg = segment_cycles(np.random.default_rng(1).standard_normal(20 * TARGET_SR).astype(np.float32))
    assert not seg.reliable


def test_short_recording_does_not_crash():
    """1,5 s : pas assez pour un rythme fiable, mais pas d'erreur."""
    seg = segment_cycles(heartbeat(0.8, 1.5))
    assert not seg.reliable


# ============================ Corrections (coupures, demi-cycle) ============================

def heartbeat_with_s2(period_s, duration_s, s2_position, s2_amplitude, noise=0.05, seed=0):
    """Variante : S2 placé à s2_position x cycle, avec l'amplitude choisie (S1 = 1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * TARGET_SR)) / TARGET_SR
    y = noise * rng.standard_normal(len(t))
    beat = 0.2
    while beat < duration_s - 0.2:
        for offset, freq, amp in [(0.0, 50, 1.0), (s2_position * period_s, 70, s2_amplitude)]:
            m = (t >= beat + offset) & (t < beat + offset + 0.08)
            y[m] += amp * np.sin(2 * np.pi * freq * t[m]) * np.hanning(m.sum())
        beat += period_s
    return ((y - y.mean()) / y.std()).astype(np.float32)


def test_silence_is_detected_and_its_cycles_rejected():
    """Une coupure de 4 s (signal nul) est repérée ; le cycle qui la contient est rejeté."""
    y = heartbeat(0.8, 20)
    y[8 * TARGET_SR:12 * TARGET_SR] = 0.0
    seg = segment_cycles(y)
    assert len(seg.silences) == 1
    start, end = seg.silences[0]
    assert start == pytest.approx(8.0, abs=0.1) and end == pytest.approx(12.0, abs=0.1)
    assert any(c.reason == "coupure" for c in seg.cycles)
    assert seg.heart_rate_bpm == pytest.approx(75, abs=4)      # le rythme n'est pas faussé


def test_no_silence_in_a_normal_recording():
    """Une diastole calme n'est pas une coupure."""
    assert segment_cycles(heartbeat(0.8, 20)).silences == []


@pytest.mark.parametrize("period", [0.9, 1.0, 1.2])
def test_half_cycle_error_is_corrected(period):
    """S2 presque aussi fort que S1, au milieu du cycle : le rythme ne doit pas être doublé."""
    seg = segment_cycles(heartbeat_with_s2(period, 20, s2_position=0.5, s2_amplitude=0.8))
    assert seg.heart_rate_bpm == pytest.approx(60 / period, rel=0.08)


def test_silence_after_zscore_is_detected():
    """Cas réel (e00120) : la coupure est nulle AVANT normalisation ; après le z-score,
    elle devient une ligne plate décalée (non nulle). Elle doit quand même être repérée."""
    rng = np.random.default_rng(3)
    t = np.arange(20 * TARGET_SR) / TARGET_SR
    raw = 0.3 + 0.05 * rng.standard_normal(len(t))           # décalage continu du micro + bruit
    for beat in np.arange(0.2, 19.8, 0.8):
        m = (t >= beat) & (t < beat + 0.08)
        raw[m] += np.sin(2 * np.pi * 50 * t[m]) * np.hanning(m.sum())
    raw[6 * TARGET_SR:10 * TARGET_SR] = 0.0                   # coupure : l'appareil n'enregistre plus
    y = ((raw - raw.mean()) / raw.std()).astype(np.float32)   # z-score : la coupure n'est plus à 0
    assert abs(y[7 * TARGET_SR]) > 0.1                        # la coupure est bien décalée
    seg = segment_cycles(y)
    assert len(seg.silences) == 1
    assert seg.silences[0][0] == pytest.approx(6.0, abs=0.1)