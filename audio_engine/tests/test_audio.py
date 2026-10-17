"""Tests unitaires du traitement audio (Service 3).

Les tests utilisent des signaux synthétiques : ils n'ont pas besoin du dataset,
ce qui permet de les exécuter partout (PC, Docker, GitHub Actions).

Lancement (depuis audio_engine/) :  pytest -v
"""
import numpy as np
import pytest
import soundfile as sf

from src.config import MIN_DURATION_S, SEGMENT_DURATION_S, TARGET_SR
from src.preprocessing import (AudioError, load_audio, normalize_audio, preprocess_file,
                               preprocess_signal, resample_audio, segment_audio,
                               validate_signal)

SEGMENT_LENGTH = int(SEGMENT_DURATION_S * TARGET_SR)


def sine(duration_s: float, sr: int, freq: float = 50.0, offset: float = 0.0) -> np.ndarray:
    """Signal de test : sinusoïde (fréquence grave, comme un bruit cardiaque)."""
    t = np.arange(int(duration_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t) + offset).astype(np.float32)


def dominant_frequency(y: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(y))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[spectrum.argmax()])


# ---------------------------- Rééchantillonnage ----------------------------

def test_resampling():
    """44100 Hz -> 2000 Hz : la durée est conservée, le nombre d'échantillons change."""
    out = resample_audio(sine(3.0, 44100), orig_sr=44100)
    assert len(out) == pytest.approx(3.0 * TARGET_SR, abs=2)


def test_resampling_keeps_heart_frequencies():
    """Une fréquence grave (50 Hz) est intacte après rééchantillonnage."""
    out = resample_audio(sine(3.0, 4000, freq=50.0), orig_sr=4000)
    assert dominant_frequency(out, TARGET_SR) == pytest.approx(50.0, abs=1.0)


def test_resampling_does_nothing_at_target_rate():
    y = sine(1.0, TARGET_SR)
    assert np.array_equal(resample_audio(y, TARGET_SR), y)


# ------------------------------ Normalisation ------------------------------

def test_normalization():
    """Après normalisation : moyenne 0 et écart-type 1, même avec un décalage continu."""
    out = normalize_audio(sine(2.0, TARGET_SR, offset=0.3))
    assert abs(out.mean()) < 1e-4
    assert out.std() == pytest.approx(1.0, abs=1e-3)


def test_normalization_rejects_silence():
    with pytest.raises(AudioError, match="silencieux"):
        normalize_audio(np.zeros(TARGET_SR * 3, dtype=np.float32))


# ------------------------------ Segmentation ------------------------------

def test_segmentation():
    """12 s -> 2 segments de 5 s (les 2 dernières secondes sont ignorées)."""
    segments = segment_audio(sine(12.0, TARGET_SR))
    assert segments.shape == (2, SEGMENT_LENGTH)


def test_segmentation_pads_short_audio_by_repetition():
    """3 s -> 1 segment de 5 s, qui commence par le signal d'origine."""
    y = sine(3.0, TARGET_SR)
    segments = segment_audio(y)
    assert segments.shape == (1, SEGMENT_LENGTH)
    assert np.allclose(segments[0, :len(y)], y)


def test_segmentation_rejects_too_short_audio():
    with pytest.raises(AudioError, match="trop court"):
        segment_audio(sine(MIN_DURATION_S - 0.5, TARGET_SR))


# ------------------------------- Robustesse -------------------------------

def test_nan_is_rejected():
    y = sine(6.0, TARGET_SR)
    y[100] = np.nan
    with pytest.raises(AudioError, match="NaN"):
        validate_signal(y)


def test_empty_signal_is_rejected():
    with pytest.raises(AudioError, match="vide"):
        validate_signal(np.array([], dtype=np.float32))


def test_stereo_signal_is_converted_to_mono():
    y = sine(6.0, TARGET_SR)
    assert validate_signal(np.stack([y, y])).shape == y.shape


def test_load_stereo_file(tmp_path):
    path = tmp_path / "stereo.wav"
    y = sine(6.0, 4000)
    sf.write(path, np.stack([y, y], axis=1), 4000)
    loaded, sr = load_audio(path)
    assert loaded.ndim == 1 and sr == 4000


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), TARGET_SR)
    with pytest.raises(AudioError):
        load_audio(path)


def test_load_wrong_format(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_text("ceci n'est pas un fichier audio")
    with pytest.raises(AudioError, match="illisible"):
        load_audio(path)


def test_load_missing_file(tmp_path):
    with pytest.raises(AudioError, match="introuvable"):
        load_audio(tmp_path / "absent.wav")


# -------------------------------- Pipeline --------------------------------

def test_preprocess_signal_output():
    """12 s à 4000 Hz -> 2 segments de 10 000 échantillons, float32, sans NaN."""
    segments = preprocess_signal(sine(12.0, 4000), sr=4000)
    assert segments.shape == (2, SEGMENT_LENGTH)
    assert segments.dtype == np.float32
    assert np.all(np.isfinite(segments))


def test_preprocess_file(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, sine(3.0, 44100), 44100)
    result = preprocess_file(path)
    assert result.padded and result.original_sr == 44100
    assert result.segments.shape == (1, SEGMENT_LENGTH)


# ========================= Caractéristiques (étape 3) =========================
from src.features import (FEATURE_NAMES, MEL_SHAPE, N_FRAMES, compute_mel_spectrogram,
                          compute_mfcc, compute_spectrogram, compute_zcr, extract_features)
from src.config import N_FFT, N_MELS, N_MFCC


@pytest.fixture
def segment() -> np.ndarray:
    """Un segment prétraité de 5 s à 2000 Hz."""
    return preprocess_signal(sine(5.0, TARGET_SR), sr=TARGET_SR)[0]


def test_mfcc(segment):
    mfcc = compute_mfcc(segment)
    assert mfcc.shape == (N_MFCC, N_FRAMES)
    assert np.all(np.isfinite(mfcc))


def test_zcr(segment):
    zcr = compute_zcr(segment)
    assert zcr.shape == (N_FRAMES,)
    assert np.all((zcr >= 0) & (zcr <= 1))


def test_zcr_is_higher_for_high_frequencies():
    """Un son aigu traverse zéro plus souvent qu'un son grave."""
    grave = compute_zcr(sine(5.0, TARGET_SR, freq=50.0))
    aigu = compute_zcr(sine(5.0, TARGET_SR, freq=500.0))
    assert aigu.mean() > grave.mean()


def test_spectrogram(segment):
    spec = compute_spectrogram(segment)
    assert spec.shape == (1 + N_FFT // 2, N_FRAMES)
    assert np.all(np.isfinite(spec))


def test_mel_spectrogram(segment):
    mel = compute_mel_spectrogram(segment)
    assert mel.shape == MEL_SHAPE == (N_MELS, N_FRAMES)
    assert np.all(np.isfinite(mel))


def test_extract_features(segment):
    vector, mel = extract_features(segment)
    assert vector.shape == (len(FEATURE_NAMES),) == (2 * N_MFCC + 2,)
    assert mel.shape == MEL_SHAPE
    assert np.all(np.isfinite(vector))