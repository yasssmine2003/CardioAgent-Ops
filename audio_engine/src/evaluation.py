"""Évaluation des modèles : métriques, Recall pathologique, sélection, matrices de confusion.

Vocabulaire (classe positive = pathologique) :
    TP : malade prédit malade        FN : malade prédit sain   <- l'erreur la plus grave
    TN : sain prédit sain            FP : sain prédit malade
    Recall pathologique = TP / (TP + FN)   (part des malades détectés)
    Spécificité         = TN / (TN + FP)   (part des sains bien reconnus)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # pas d'affichage : les figures sont seulement sauvegardées
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.config import DECISION_THRESHOLD, MIN_SPECIFICITY

CLASS_NAMES = ["normal", "pathologique"]


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float = DECISION_THRESHOLD) -> dict[str, float]:
    """Toutes les métriques à partir des probabilités de la classe pathologique."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # moyennes sur les 2 classes (vue d'ensemble)
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # classe pathologique (critère imposé par le cahier des charges)
        "recall_pathological": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision_pathological": tp / (tp + fp) if (tp + fp) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: float(v) for k, v in metrics.items()}


def aggregate_by_recording(meta: pd.DataFrame, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Passe des segments aux enregistrements : moyenne des probabilités de chaque audio.

    Returns:
        (labels des enregistrements, probabilités des enregistrements)
    """
    df = pd.DataFrame({"path": meta["path"], "label": meta["label"], "proba": y_proba})
    per_recording = df.groupby("path").agg(label=("label", "first"), proba=("proba", "mean"))
    return per_recording["label"].to_numpy(), per_recording["proba"].to_numpy()


def selection_score(metrics: dict[str, float], min_specificity: float = MIN_SPECIFICITY) -> float:
    """Critère de sélection : le Recall pathologique, à condition que la spécificité soit suffisante.

    Un modèle qui répond toujours « pathologique » a un Recall de 1 mais est inutile :
    s'il ne reconnaît pas au moins `min_specificity` des normaux, son score devient négatif
    et il est classé derrière tous les modèles valides.
    """
    if metrics["specificity"] >= min_specificity:
        return metrics["recall_pathological"]
    return metrics["specificity"] - 1.0


def plot_confusion_matrix(y_true: np.ndarray, y_proba: np.ndarray, title: str,
                          threshold: float = DECISION_THRESHOLD) -> plt.Figure:
    """Matrice de confusion annotée (nombres et pourcentages par ligne)."""
    cm = confusion_matrix(y_true, (y_proba >= threshold).astype(int), labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f} %)", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Prédit", ylabel="Réel", title=title)
    fig.tight_layout()
    return fig