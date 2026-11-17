"""API interne du Service 3 (audio-engine), appelée par le Service 4 (agent-orchestrator).

    GET  /health   -> le service est-il prêt ? (modèle chargé)
    POST /predict  -> fichier .wav (multipart, champ « file ») -> PredictionResult (JSON)

Lancement local (depuis audio_engine/) :  uvicorn src.api:app --port 8000
Documentation automatique : http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src import config
from src.inference import CardioPredictor
from src.preprocessing import AudioError
from src.schemas import PredictionResult

logger = logging.getLogger("audio_engine.api")
MAX_UPLOAD_MB = float(os.getenv("AUDIO_ENGINE_MAX_UPLOAD_MB", "20"))   # [C] protège la mémoire du service
ALLOWED_SUFFIXES = {".wav"}                                            # [A] entrée du service : audio .wav


def warm_up(predictor: CardioPredictor) -> None:
    """Une première prédiction à vide : la 1re vraie requête n'est pas plus lente que les autres."""
    rng = np.random.default_rng(0)
    predictor.predict_signal(rng.standard_normal(int(config.SEGMENT_DURATION_S * config.TARGET_SR))
                             .astype(np.float32), config.TARGET_SR)


def create_app(predictor_factory: Callable[[], CardioPredictor] = CardioPredictor) -> FastAPI:
    """Fabrique l'application ; les tests y passent un faux modèle."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor, app.state.error = None, None
        try:
            predictor = predictor_factory()          # chargé UNE fois au démarrage
            warm_up(predictor)
            app.state.predictor = predictor
            logger.info("Modèle prêt : %s", predictor.card["model_name"])
        except Exception as exc:                     # le service démarre, mais /health dira pourquoi
            app.state.error = str(exc)
            logger.error("Modèle non chargé : %s", exc)
        yield

    app = FastAPI(title="CardioAgent-Ops — audio-engine", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail=f"Modèle non chargé : {app.state.error}")
        return {"status": "ok", "model": predictor.card["model_name"], "threshold": predictor.threshold}

    @app.post("/predict", response_model=PredictionResult)
    async def predict(file: UploadFile = File(..., description="Enregistrement cardiaque .wav")) -> PredictionResult:
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        if Path(file.filename or "").suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format accepté : .wav")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")
        if len(content) > MAX_UPLOAD_MB * 1e6:
            raise HTTPException(status_code=413, detail=f"Fichier > {MAX_UPLOAD_MB:g} Mo")

        # Fichier temporaire : on réutilise exactement le chargement de l'entraînement
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            path.write_bytes(content)
            try:
                return await run_in_threadpool(predictor.predict, path)   # calcul hors de la boucle async
            except AudioError as exc:                                     # trop court, silencieux...
                raise HTTPException(status_code=422, detail=f"Audio inutilisable : {exc}") from exc
            except Exception as exc:                                      # fichier illisible
                logger.warning("Lecture impossible (%s) : %s", file.filename, exc)
                raise HTTPException(status_code=400, detail="Fichier .wav illisible") from exc

    return app


app = create_app()
