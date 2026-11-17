"""Tests de l'API interne (FastAPI + client HTTPX), avec le faux modèle ONNX des tests d'inférence."""
import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api import create_app
from src.inference import CardioPredictor
from tests.test_inference import make_fake_model


def wav_bytes(seconds: float, sr: int = 4000) -> bytes:
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    sf.write(buffer, rng.standard_normal(int(seconds * sr)).astype(np.float32) * 0.1, sr, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    model_path, card_path = make_fake_model(tmp_path)
    with TestClient(create_app(lambda: CardioPredictor(model_path, card_path))) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_valid_json(client):
    r = client.post("/predict", files={"file": ("coeur.wav", wav_bytes(12), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"normal", "pathologique"}
    assert body["n_segments"] == 2 and body["original_sample_rate"] == 4000


@pytest.mark.parametrize("name, content, status", [
    ("coeur.mp3", b"xx", 415),             # mauvais format
    ("coeur.wav", b"", 400),               # fichier vide
    ("coeur.wav", b"pas un wav", 422),     # fichier illisible
])
def test_bad_files_are_rejected(client, name, content, status):
    assert client.post("/predict", files={"file": (name, content, "audio/wav")}).status_code == status


def test_too_short_audio_is_422(client):
    r = client.post("/predict", files={"file": ("court.wav", wav_bytes(1), "audio/wav")})
    assert r.status_code == 422


def test_missing_model_gives_503(tmp_path):
    with TestClient(create_app(lambda: CardioPredictor(tmp_path / "absent.onnx", tmp_path / "absent.json"))) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", files={"file": ("a.wav", wav_bytes(6), "audio/wav")}).status_code == 503
