"""Tests des métriques et du critère de sélection (sans dataset)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import aggregate_by_recording, compute_metrics, selection_score


def test_recall_pathological():
    """4 malades dont 3 détectés -> Recall pathologique = 3 / (3 + 1) = 0,75."""
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.6])
    m = compute_metrics(y_true, y_proba)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 1, 1)
    assert m["recall_pathological"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.5)


def test_selection_rejects_trivial_model():
    """Répondre toujours « pathologique » donne un Recall de 1, mais le modèle doit être écarté."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    always_sick = compute_metrics(y_true, np.ones(6))
    reasonable = compute_metrics(y_true, np.array([0.9, 0.4, 0.1, 0.2, 0.3, 0.1]))
    assert always_sick["recall_pathological"] == 1.0
    assert selection_score(always_sick) < 0 < selection_score(reasonable)


def test_aggregate_by_recording():
    """Les probabilités des segments d'un même audio sont moyennées."""
    meta = pd.DataFrame({"path": ["a", "a", "b"], "label": [1, 1, 0]})
    labels, proba = aggregate_by_recording(meta, np.array([0.2, 0.8, 0.1]))
    assert labels.tolist() == [1, 0]
    assert proba.tolist() == pytest.approx([0.5, 0.1])