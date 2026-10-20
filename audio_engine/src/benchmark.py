"""Benchmark des 5 modèles : comparaison à spécificité égale et sélection du meilleur.

Principe :
1. pour chaque modèle, on récupère dans MLflow son run parent le plus récent et le modèle enregistré ;
2. on choisit son SEUIL sur la validation : le plus grand Recall pathologique possible
   avec une spécificité >= MIN_SPECIFICITY (tous les modèles sont jugés à la même exigence) ;
3. on évalue sur le test avec ce seuil, on calcule l'AUC et on mesure l'inférence sur CPU ;
4. le meilleur modèle est choisi sur la VALIDATION (le test sert seulement à rapporter).
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src import config
from src.evaluation import aggregate_by_recording, compute_metrics
from src.feature_loader import load_1d, load_mel

logger = logging.getLogger(__name__)
CPU_REPEATS = 100  # nombre de prédictions pour mesurer le temps d'inférence sur CPU
GPU_BATCH = 256    # taille du lot pour mesurer la mémoire et le temps d'inférence sur GPU
GPU_REPEATS = 20


@dataclass
class LoadedModel:
    name: str
    family: str          # "classical" ou "deep"
    run_id: str
    model: object
    run_metrics: dict


# ======================= 1. Récupération des modèles =======================

def find_latest_run(model_name: str) -> pd.Series:
    """Run parent le plus récent d'un modèle (seuls les runs parents portent le tag model_name)."""
    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],
                              filter_string=f"tags.model_name = '{model_name}'",
                              order_by=["start_time DESC"])
    if runs.empty:
        raise LookupError(f"Aucun run MLflow pour le modèle « {model_name} »")
    return runs.iloc[0]


def load_model(model_name: str) -> LoadedModel:
    """Charge le modèle enregistré dans MLflow (sur CPU pour les réseaux)."""
    run = find_latest_run(model_name)
    family = run["tags.family"]
    uri = f"runs:/{run['run_id']}/model"
    if family == "classical":
        model = mlflow.sklearn.load_model(uri)
    else:
        import torch
        model = mlflow.pytorch.load_model(uri, map_location=torch.device("cpu")).eval()
    metrics = {k.removeprefix("metrics."): v for k, v in run.items() if k.startswith("metrics.")}
    return LoadedModel(model_name, family, run["run_id"], model, metrics)


# ========================= 2. Prédictions =========================

def predict_proba(loaded: LoadedModel, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Probabilité « pathologique » de chaque segment, quel que soit le type de modèle."""
    if loaded.family == "classical":
        return loaded.model.predict_proba(X)[:, 1]
    import torch
    with torch.no_grad():
        scores = [loaded.model(torch.from_numpy(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.sigmoid(torch.cat(scores)).numpy()


def load_inputs(family: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Vecteurs 1D pour les modèles classiques, Mel-spectrogrammes pour les réseaux."""
    return load_1d(split) if family == "classical" else load_mel(split)


# ======================= 3. Seuil et métriques =======================

def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_specificity: float) -> float:
    """Seuil qui donne le plus grand Recall pathologique avec une spécificité >= min_specificity.

    À Recall égal, on garde le seuil le plus exigeant (meilleure spécificité).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = (1 - fpr) >= min_specificity
    best = np.flatnonzero(valid & (tpr == tpr[valid].max()))[0]   # 1er = plus faible taux de FP
    return float(min(thresholds[best], 1.0))


def measure_cpu_latency_ms(loaded: LoadedModel, X: np.ndarray) -> float:
    """Temps médian pour prédire UN segment sur CPU (conditions proches du conteneur Docker)."""
    one = X[:1]
    predict_proba(loaded, one)  # échauffement
    times = []
    for _ in range(CPU_REPEATS):
        start = time.perf_counter()
        predict_proba(loaded, one)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def measure_gpu_inference(loaded: LoadedModel, X: np.ndarray) -> tuple[float, float]:
    """Mémoire GPU maximale (Mo) et temps par segment (ms) pour prédire un lot de GPU_BATCH segments.

    Mesure identique pour tous les réseaux : seuls le modèle et le lot sont sur le GPU.
    Retourne (NaN, NaN) pour les modèles scikit-learn (ils ne s'exécutent pas sur GPU)
    ou si aucun GPU n'est disponible.
    """
    if loaded.family == "classical":
        return float("nan"), float("nan")
    import torch
    if not torch.cuda.is_available():
        return float("nan"), float("nan")

    torch.cuda.empty_cache()
    model = loaded.model.to("cuda")
    batch = torch.from_numpy(np.ascontiguousarray(X[:GPU_BATCH])).to("cuda")
    with torch.no_grad():
        model(batch)                              # échauffement
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(GPU_REPEATS):
            model(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6  # poids + lot + calculs intermédiaires
    loaded.model = model.to("cpu")
    del batch
    torch.cuda.empty_cache()
    return peak_mb, elapsed / GPU_REPEATS / len(X[:GPU_BATCH]) * 1000


def model_size_mb(loaded: LoadedModel) -> float:
    """Taille du modèle en mémoire (Mo) : poids des réseaux ou objet sklearn sérialisé."""
    if loaded.family == "classical":
        return len(pickle.dumps(loaded.model)) / 1e6
    tensors = list(loaded.model.parameters()) + list(loaded.model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


def evaluate_model(loaded: LoadedModel) -> tuple[dict, dict]:
    """Évalue un modèle à spécificité égale. Retourne (ligne du tableau, probabilités pour les ROC)."""
    X_val, y_val, _ = load_inputs(loaded.family, "val")
    X_test, y_test, meta_test = load_inputs(loaded.family, "test")
    p_val, p_test = predict_proba(loaded, X_val), predict_proba(loaded, X_test)

    threshold = tune_threshold(y_val, p_val, config.MIN_SPECIFICITY)
    val = compute_metrics(y_val, p_val, threshold)
    test = compute_metrics(y_test, p_test, threshold)
    rec_test = compute_metrics(*aggregate_by_recording(meta_test, p_test), threshold)
    alt_threshold = tune_threshold(y_val, p_val, config.ALT_SPECIFICITY)
    test_alt = compute_metrics(y_test, p_test, alt_threshold)

    gpu_memory, gpu_ms = measure_gpu_inference(loaded, X_test)
    row = {
        "model": loaded.name,
        "threshold": threshold,
        "val_recall_pathological": val["recall_pathological"],
        "val_specificity": val["specificity"],
        "val_auc": roc_auc_score(y_val, p_val),
        "accuracy": test["accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "recall_pathological": test["recall_pathological"],
        "specificity": test["specificity"],
        "auc": roc_auc_score(y_test, p_test),
        "missed_pathological": int(test["fn"]),
        "false_alarms": int(test["fp"]),
        "rec_recall_pathological": rec_test["recall_pathological"],
        "rec_specificity": rec_test["specificity"],
        f"recall_pathological_at_spec{int(config.ALT_SPECIFICITY * 100)}": test_alt["recall_pathological"],
        "training_time_s": loaded.run_metrics.get("training_time_s", np.nan),
        "inference_time_ms_cpu": measure_cpu_latency_ms(loaded, X_test),
        "model_size_mb": model_size_mb(loaded),
        "gpu_inference_memory_mb": gpu_memory,
        "gpu_inference_ms_per_segment": gpu_ms,
        # mesure d'entraînement, conservée pour mémoire : non comparable entre modèles
        "train_gpu_memory_peak_mb": loaded.run_metrics.get("gpu_memory_peak_mb", np.nan),
        "run_id": loaded.run_id,
        "family": loaded.family,
    }
    return row, {"y_test": y_test, "p_test": p_test, "threshold": threshold}


# ======================= 4. Benchmark et sélection =======================

def benchmark_models(model_names=config.BENCHMARK_MODELS) -> tuple[pd.DataFrame, dict]:
    """Évalue tous les modèles. Retourne (tableau du benchmark, données des courbes ROC)."""
    rows, curves = [], {}
    for name in model_names:
        logger.info("Évaluation de %s", name)
        row, curve = evaluate_model(load_model(name))
        rows.append(row)
        curves[name] = curve
    return pd.DataFrame(rows), curves


def select_best_model(results: pd.DataFrame) -> dict:
    """Meilleur modèle : plus grand Recall pathologique sur la VALIDATION (spécificité >= seuil),
    puis plus grande AUC de validation en cas d'égalité. Le test n'intervient pas dans le choix."""
    ranked = results.sort_values(["val_recall_pathological", "val_auc"], ascending=False)
    best = ranked.iloc[0]
    return {
        "model_name": best["model"],
        "family": best["family"],
        "run_id": best["run_id"],
        "model_uri": f"runs:/{best['run_id']}/model",
        "threshold": float(best["threshold"]),
        "selection_rule": (f"max Recall pathologique (validation) avec spécificité >= "
                           f"{config.MIN_SPECIFICITY}, puis AUC de validation"),
        "test_metrics": {k: float(best[k]) for k in
                         ["recall_pathological", "specificity", "accuracy", "f1", "auc"]},
    }


def plot_roc_curves(curves: dict, best_name: str) -> plt.Figure:
    """Courbes ROC des 5 modèles sur le test, avec le point de fonctionnement choisi."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, c in curves.items():
        fpr, tpr, _ = roc_curve(c["y_test"], c["p_test"])
        auc = roc_auc_score(c["y_test"], c["p_test"])
        ax.plot(1 - fpr, tpr, lw=2.5 if name == best_name else 1.2, label=f"{name} (AUC {auc:.3f})")
        y_pred = c["p_test"] >= c["threshold"]
        spec = ((~y_pred) & (c["y_test"] == 0)).sum() / (c["y_test"] == 0).sum()
        rec = (y_pred & (c["y_test"] == 1)).sum() / (c["y_test"] == 1).sum()
        ax.plot(spec, rec, "o", color=ax.lines[-1].get_color())
    ax.axvline(config.MIN_SPECIFICITY, ls="--", color="grey", lw=0.8)
    ax.set(xlabel="Spécificité", ylabel="Recall pathologique", xlim=(1, 0.4), ylim=(0.4, 1.01),
           title="Courbes ROC (test) et seuils choisis sur la validation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig