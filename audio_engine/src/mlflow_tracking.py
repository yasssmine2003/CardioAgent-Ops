"""Traçabilité MLflow : configuration et enregistrement des métriques et figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow

from src import config


def setup_mlflow(experiment: str = config.MLFLOW_EXPERIMENT) -> None:
    """Indique à MLflow où écrire (dossier local, Google Drive ou serveur mlflow-server)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)


def log_preprocessing_params() -> None:
    """Paramètres communs à tous les modèles : prétraitement et caractéristiques."""
    mlflow.log_params({
        "sample_rate": config.TARGET_SR,
        "segment_duration_s": config.SEGMENT_DURATION_S,
        "n_mfcc": config.N_MFCC,
        "n_mels": config.N_MELS,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "decision_threshold": config.DECISION_THRESHOLD,
        "min_specificity": config.MIN_SPECIFICITY,
        "seed": config.RANDOM_SEED,
    })


def log_metrics(metrics: dict[str, float], prefix: str) -> None:
    """Enregistre un dictionnaire de métriques avec un préfixe (val_, test_, rec_test_...)."""
    mlflow.log_metrics({f"{prefix}{name}": value for name, value in metrics.items()})


def log_figure(fig: plt.Figure, filename: str) -> None:
    mlflow.log_figure(fig, filename)
    plt.close(fig)