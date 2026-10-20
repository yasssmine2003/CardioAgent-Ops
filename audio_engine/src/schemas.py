"""Format de sortie du Service 3, validé avec Pydantic (v2).

Le cahier des charges impose la sortie « Probabilités & Caractéristiques » vers l'agent-orchestrator
mais pas son format exact : ce schéma est une PROPOSITION d'intégration (à valider avec le binôme).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    """Caractéristiques résumées de l'enregistrement (moyennes sur ses segments)."""
    mfcc_mean: list[float] = Field(..., min_length=1, description="Moyenne de chaque coefficient MFCC")
    zcr_mean: float = Field(..., ge=0, le=1, description="Taux moyen de passages par zéro")


class ModelInfo(BaseModel):
    name: str
    mlflow_run_id: str
    threshold: float = Field(..., ge=0, le=1)
    decision_level: str


class PredictionResult(BaseModel):
    """Réponse du Service 3 pour un enregistrement."""
    prediction: Literal["normal", "pathologique"]
    probability: float = Field(..., ge=0, le=1, description="Probabilité « pathologique » de l'enregistrement")
    confidence: float = Field(..., ge=0, le=1, description="Probabilité de la classe prédite")
    segment_probabilities: list[float] = Field(..., min_length=1)
    n_segments: int = Field(..., ge=1)
    duration_s: float = Field(..., gt=0)
    original_sample_rate: int = Field(..., gt=0)
    padded: bool = Field(..., description="Audio de 2 à 5 s complété par répétition")
    features: AudioFeatures
    model: ModelInfo
    sample_rate: int
    segment_duration_s: float
    inference_time_ms: float = Field(..., ge=0)