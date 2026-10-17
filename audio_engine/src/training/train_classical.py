"""Entraînement des modèles classiques (Decision Tree, SVM) avec suivi MLflow.

Pour chaque modèle :
1. chaque combinaison d'hyperparamètres est entraînée sur le train et évaluée sur la validation
   (un run MLflow « enfant » par combinaison) ;
2. la meilleure combinaison (critère : Recall pathologique sous contrainte de spécificité)
   est évaluée sur le test, et enregistrée dans le run « parent » avec le modèle.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np

from src.evaluation import (aggregate_by_recording, compute_metrics, plot_confusion_matrix,
                            selection_score)
from src.feature_loader import load_1d
from src.mlflow_tracking import log_figure, log_metrics, log_preprocessing_params
from src.models import CLASSICAL_MODELS

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    params: dict
    model: object
    val_metrics: dict
    score: float
    train_time_s: float


def fit_and_validate(module, params: dict, X_train, y_train, X_val, y_val) -> Candidate:
    """Entraîne une combinaison d'hyperparamètres et l'évalue sur la validation."""
    model = module.build_model(**params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    val_metrics = compute_metrics(y_val, model.predict_proba(X_val)[:, 1])
    return Candidate(params, model, val_metrics, selection_score(val_metrics), train_time)


def measure_inference_ms(model, X: np.ndarray) -> float:
    """Temps moyen de prédiction pour UN segment, en millisecondes."""
    start = time.perf_counter()
    model.predict_proba(X)
    return (time.perf_counter() - start) / len(X) * 1000


def run_classical(model_name: str) -> dict[str, float]:
    """Entraîne, sélectionne, évalue et enregistre un modèle classique. Retourne ses métriques."""
    module = CLASSICAL_MODELS[model_name]
    X_train, y_train, _ = load_1d("train")
    X_val, y_val, meta_val = load_1d("val")
    X_test, y_test, meta_test = load_1d("test")

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({"model_name": model_name, "family": "classical", "input": "features_1d"})
        log_preprocessing_params()

        best = None
        for params in module.PARAM_GRID:
            with mlflow.start_run(run_name=f"{model_name} {params}", nested=True):
                candidate = fit_and_validate(module, params, X_train, y_train, X_val, y_val)
                mlflow.log_params(params)
                log_metrics(candidate.val_metrics, "val_")
                mlflow.log_metrics({"selection_score": candidate.score,
                                    "training_time_s": candidate.train_time_s})
            logger.info("%s %s -> recall_patho=%.3f spécificité=%.3f", model_name, params,
                        candidate.val_metrics["recall_pathological"],
                        candidate.val_metrics["specificity"])
            if best is None or candidate.score > best.score:
                best = candidate

        # ----- Meilleure combinaison : évaluation finale et enregistrement -----
        test_proba = best.model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba)
        rec_val = compute_metrics(*aggregate_by_recording(meta_val, best.model.predict_proba(X_val)[:, 1]))
        rec_test = compute_metrics(*aggregate_by_recording(meta_test, test_proba))

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        log_metrics(best.val_metrics, "val_")
        log_metrics(test_metrics, "test_")
        log_metrics(rec_val, "rec_val_")
        log_metrics(rec_test, "rec_test_")
        mlflow.log_metrics({"selection_score": best.score,
                            "training_time_s": best.train_time_s,
                            "inference_time_ms": measure_inference_ms(best.model, X_test)})
        log_figure(plot_confusion_matrix(y_test, test_proba, f"{model_name} — test (segments)"),
                   "confusion_matrix_test.png")
        mlflow.sklearn.log_model(best.model, artifact_path="model", input_example=X_test[:2])

    return {"model": model_name, **{f"val_{k}": v for k, v in best.val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()}}