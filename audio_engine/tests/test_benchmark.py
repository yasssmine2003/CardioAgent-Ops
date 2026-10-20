"""Tests du choix de seuil et de la sélection du meilleur modèle (sans MLflow ni dataset)."""
import numpy as np
import pandas as pd
import pytest

from src.benchmark import select_best_model, tune_threshold
from src.evaluation import compute_metrics


def test_tune_threshold_respects_min_specificity():
    """Le seuil choisi garde au moins 70 % de spécificité et maximise le Recall."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 60)
    proba = np.concatenate([rng.uniform(0, 0.7, 200), rng.uniform(0.3, 1, 60)])
    threshold = tune_threshold(y, proba, min_specificity=0.70)
    m = compute_metrics(y, proba, threshold)
    assert m["specificity"] >= 0.70
    # un seuil un peu plus bas ferait passer la spécificité sous 70 % (ou ne gagnerait rien)
    lower = compute_metrics(y, proba, threshold - 1e-6)
    assert lower["specificity"] < 0.70 or lower["recall_pathological"] == m["recall_pathological"]


def test_select_best_model_uses_validation_not_test():
    """Le choix se fait sur la validation, même si un autre modèle est meilleur sur le test."""
    results = pd.DataFrame([
        {"model": "a", "family": "deep", "run_id": "1", "threshold": 0.4,
         "val_recall_pathological": 0.95, "val_auc": 0.90, "recall_pathological": 0.90,
         "specificity": 0.75, "accuracy": 0.8, "f1": 0.8, "auc": 0.9},
        {"model": "b", "family": "classical", "run_id": "2", "threshold": 0.5,
         "val_recall_pathological": 0.90, "val_auc": 0.95, "recall_pathological": 0.99,
         "specificity": 0.75, "accuracy": 0.8, "f1": 0.8, "auc": 0.9},
    ])
    best = select_best_model(results)
    assert best["model_name"] == "a"
    assert best["threshold"] == pytest.approx(0.4)


def test_gpu_measure_not_applicable_to_classical_models():
    """Les modèles scikit-learn ne s'exécutent pas sur GPU : la mesure vaut NaN."""
    from src.benchmark import LoadedModel, measure_gpu_inference
    loaded = LoadedModel("svm", "classical", "run", model=None, run_metrics={})
    memory, ms = measure_gpu_inference(loaded, np.zeros((4, 28), dtype=np.float32))
    assert np.isnan(memory) and np.isnan(ms)