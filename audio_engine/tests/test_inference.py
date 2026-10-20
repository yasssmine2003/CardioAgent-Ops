"""Tests de l'inférence ONNX Runtime (sans PyTorch).

Un petit modèle ONNX « factice » de même entrée et même sortie que le vrai CNN est construit
à la volée : les tests ne dépendent ni du dataset ni de l'entraînement (exécutables en CI).
"""
import json

import numpy as np
import pytest
import soundfile as sf

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper

from src import config
from src.features import MEL_SHAPE
from src.inference import CardioPredictor, ModelConfigError
from src.preprocessing import AudioError


def make_fake_model(tmp_path, threshold=0.5, **settings_override):
    """Modèle ONNX factice : probabilité = sigmoïde(moyenne du Mel-spectrogramme)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("mel_spectrogram", TensorProto.FLOAT, ["batch", 1, *MEL_SHAPE])
    p = helper.make_tensor_value_info("probability", TensorProto.FLOAT, ["batch"])
    nodes = [helper.make_node("ReduceMean", ["mel_spectrogram"], ["m"], axes=[1, 2, 3], keepdims=0),
             helper.make_node("Sigmoid", ["m"], ["probability"])]
    model = helper.make_model(helper.make_graph(nodes, "fake", [x], [p]),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # version lisible par toutes les versions récentes d'ONNX Runtime
    model_path, card_path = tmp_path / "model.onnx", tmp_path / "model.json"
    onnx.save(model, model_path)
    preprocessing = {"sample_rate": config.TARGET_SR, "segment_duration_s": config.SEGMENT_DURATION_S,
                     "n_fft": config.N_FFT, "hop_length": config.HOP_LENGTH, "n_mels": config.N_MELS}
    preprocessing.update(settings_override)
    card = {"model_name": "fake", "mlflow_run_id": "test", "input_name": "mel_spectrogram",
            "output_name": "probability", "preprocessing": preprocessing,
            "decision": {"threshold": threshold, "level": "enregistrement"}}
    card_path.write_text(json.dumps(card))
    return model_path, card_path


def write_heart_like_wav(path, duration_s=12.0, sr=4000):
    t = np.arange(int(duration_s * sr)) / sr
    sf.write(path, (0.5 * np.sin(2 * np.pi * 50 * t)).astype(np.float32), sr)
    return path


def test_onnx_inference_output(tmp_path):
    """Un .wav de 12 s -> 2 segments, une réponse complète et valide."""
    predictor = CardioPredictor(*make_fake_model(tmp_path))
    result = predictor.predict(write_heart_like_wav(tmp_path / "a.wav"))
    assert result.n_segments == len(result.segment_probabilities) == 2
    assert 0 <= result.probability <= 1
    assert result.prediction == ("pathologique" if result.probability >= 0.5 else "normal")
    assert len(result.features.mfcc_mean) == config.N_MFCC
    assert result.original_sample_rate == 4000 and result.sample_rate == config.TARGET_SR


def test_threshold_drives_the_decision(tmp_path):
    """Seuil 0 -> toujours pathologique ; seuil 1 -> toujours normal."""
    wav = write_heart_like_wav(tmp_path / "a.wav")
    low = CardioPredictor(*make_fake_model(tmp_path / "low", threshold=0.0))
    high = CardioPredictor(*make_fake_model(tmp_path / "high", threshold=1.0))
    assert low.predict(wav).prediction == "pathologique"
    assert high.predict(wav).prediction == "normal"


def test_incompatible_model_card_is_rejected(tmp_path):
    """Un modèle entraîné avec d'autres réglages (ici 4000 Hz) est refusé."""
    with pytest.raises(ModelConfigError):
        CardioPredictor(*make_fake_model(tmp_path, sample_rate=4000))


def test_too_short_audio_is_rejected(tmp_path):
    predictor = CardioPredictor(*make_fake_model(tmp_path))
    with pytest.raises(AudioError):
        predictor.predict(write_heart_like_wav(tmp_path / "short.wav", duration_s=1.0))


@pytest.mark.skipif(not config.ONNX_MODEL_PATH.is_file(), reason="cardio_model.onnx absent")
def test_deployed_model_loads_and_predicts(tmp_path):
    """Le vrai modèle déployé se charge et produit une sortie de la forme attendue."""
    result = CardioPredictor().predict(write_heart_like_wav(tmp_path / "a.wav"))
    assert result.model.name and 0 <= result.probability <= 1