"""Entraînement des modèles de deep learning (CNN, ResNet18, Transformer-CNN) avec suivi MLflow.

Même logique que les modèles classiques :
1. chaque combinaison d'hyperparamètres = un run enfant, entraîné sur le train,
   arrêté au meilleur score de validation (early stopping) ;
2. la meilleure combinaison est évaluée sur le test et enregistrée dans le run parent.
"""
from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from torch import nn

from src import config
from src.evaluation import (aggregate_by_recording, compute_metrics, plot_confusion_matrix,
                            selection_score)
from src.feature_loader import load_mel
from src.mlflow_tracking import log_figure, log_metrics, log_preprocessing_params
from src.models import DEEP_MODELS
from src.models.common import NormalizedModel, count_parameters

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DeepCandidate:
    params: dict
    model: nn.Module
    val_metrics: dict
    score: float
    train_time_s: float
    best_epoch: int
    history: dict


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Fixe tous les générateurs aléatoires : deux entraînements identiques donnent le même modèle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """Probabilité « pathologique » de chaque segment (sigmoïde du score du réseau)."""
    model.eval()
    scores = [model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.sigmoid(torch.cat(scores)).cpu().numpy()


def train_one_epoch(model, X, y, optimizer, loss_fn, generator) -> float:
    """Un passage complet sur le train, par lots mélangés. Retourne la perte moyenne."""
    model.train()
    order = torch.randperm(len(X), generator=generator, device="cpu").to(X.device)
    total = 0.0
    for i in range(0, len(X), config.BATCH_SIZE):
        idx = order[i:i + config.BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)


def fit_and_validate(module, params: dict, data: dict, norm: tuple[float, float]) -> DeepCandidate:
    """Entraîne une combinaison ; garde l'époque au meilleur score de validation."""
    set_seed()
    model = NormalizedModel(module.build_model(**params), *norm).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]  # couches figées exclues
    optimizer = torch.optim.Adam(trainable, lr=params["learning_rate"],
                                 weight_decay=config.WEIGHT_DECAY)
    # Rééquilibrage : une erreur sur un malade pèse (nb normaux / nb malades) fois plus
    n_pos = float(data["y_train"].sum())
    pos_weight = torch.tensor((len(data["y_train"]) - n_pos) / n_pos, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)

    history = {"train_loss": [], "val_loss": [], "val_recall_pathological": [], "val_specificity": []}
    best_score, best_state, best_epoch = -np.inf, None, 0
    best_val_loss, waited = np.inf, 0
    start = time.perf_counter()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, data["X_train"], data["y_train"], optimizer, loss_fn, generator)
        with torch.no_grad():
            model.eval()
            val_loss = loss_fn(model(data["X_val"]), data["y_val"]).item()
        val_metrics = compute_metrics(data["y_val_np"], predict_proba(model, data["X_val"]))
        score = selection_score(val_metrics)

        for key, value in [("train_loss", train_loss), ("val_loss", val_loss),
                           ("val_recall_pathological", val_metrics["recall_pathological"]),
                           ("val_specificity", val_metrics["specificity"])]:
            history[key].append(value)
            mlflow.log_metric(key, value, step=epoch)

        # 1) Meilleure époque = meilleur score selon le critère du cahier des charges
        if score > best_score:
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        # 2) Arrêt anticipé = quand la perte de validation ne baisse plus (le modèle n'apprend plus)
        if val_loss < best_val_loss:
            best_val_loss, waited = val_loss, 0
        else:
            waited += 1
            if waited >= config.PATIENCE:
                break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    val_metrics = compute_metrics(data["y_val_np"], predict_proba(model, data["X_val"]))
    return DeepCandidate(params, model, val_metrics, best_score, train_time, best_epoch, history)


def plot_history(history: dict, best_epoch: int, title: str) -> plt.Figure:
    """Courbes d'apprentissage : perte (train / validation) et Recall / spécificité de validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(title="Perte", xlabel="Époque")
    ax2.plot(epochs, history["val_recall_pathological"], label="Recall pathologique")
    ax2.plot(epochs, history["val_specificity"], label="Spécificité")
    ax2.axhline(config.MIN_SPECIFICITY, ls="--", color="grey", lw=0.8)
    ax2.set(title="Validation", xlabel="Époque", ylim=(0, 1))
    for ax in (ax1, ax2):
        ax.axvline(best_epoch, ls=":", color="black", lw=0.8)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def load_data() -> dict:
    """Charge les Mel-spectrogrammes des 3 ensembles et les place sur le GPU."""
    data = {}
    for split in ("train", "val", "test"):
        X, y, meta = load_mel(split)
        data[f"X_{split}"] = torch.from_numpy(X).to(DEVICE)
        data[f"y_{split}"] = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
        data[f"y_{split}_np"] = y
        data[f"meta_{split}"] = meta
    return data


def run_deep(model_name: str) -> dict[str, float]:
    """Entraîne, sélectionne, évalue et enregistre un modèle de deep learning."""
    module = DEEP_MODELS[model_name]
    data = load_data()
    # Statistiques de standardisation calculées sur le TRAIN uniquement
    norm = (data["X_train"].mean().item(), data["X_train"].std().item())
    logger.info("Appareil : %s | standardisation : moyenne=%.2f, écart-type=%.2f", DEVICE, *norm)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({"model_name": model_name, "family": "deep", "input": "mel_spectrogram",
                         "device": str(DEVICE)})
        log_preprocessing_params()
        mlflow.log_params({"batch_size": config.BATCH_SIZE, "max_epochs": config.MAX_EPOCHS,
                           "patience": config.PATIENCE, "weight_decay": config.WEIGHT_DECAY,
                           "norm_mean": round(norm[0], 4), "norm_std": round(norm[1], 4)})

        best = None
        for params in module.PARAM_GRID:
            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with mlflow.start_run(run_name=f"{model_name} {params}", nested=True):
                mlflow.log_params(params)
                candidate = fit_and_validate(module, params, data, norm)
                log_metrics(candidate.val_metrics, "best_val_")
                mlflow.log_metrics({"selection_score": candidate.score, "best_epoch": candidate.best_epoch,
                                    "training_time_s": candidate.train_time_s})
                log_figure(plot_history(candidate.history, candidate.best_epoch, f"{model_name} {params}"),
                           "learning_curves.png")
            logger.info("%s %s -> époque %d, recall_patho=%.3f spécificité=%.3f", model_name, params,
                        candidate.best_epoch, candidate.val_metrics["recall_pathological"],
                        candidate.val_metrics["specificity"])
            if best is None or candidate.score > best.score:
                best = candidate

        # ----- Meilleure combinaison : évaluation finale et enregistrement -----
        model = best.model
        test_proba = predict_proba(model, data["X_test"])
        val_proba = predict_proba(model, data["X_val"])
        test_metrics = compute_metrics(data["y_test_np"], test_proba)
        start = time.perf_counter()
        predict_proba(model, data["X_test"])
        inference_ms = (time.perf_counter() - start) / len(data["X_test"]) * 1000

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        log_metrics(best.val_metrics, "val_")
        log_metrics(test_metrics, "test_")
        log_metrics(compute_metrics(*aggregate_by_recording(data["meta_val"], val_proba)), "rec_val_")
        log_metrics(compute_metrics(*aggregate_by_recording(data["meta_test"], test_proba)), "rec_test_")
        mlflow.log_metrics({"selection_score": best.score, "training_time_s": best.train_time_s,
                            "best_epoch": best.best_epoch, "inference_time_ms": inference_ms,
                            "n_parameters": count_parameters(model),
                            "n_trainable_parameters": count_parameters(model, trainable_only=True)})
        if DEVICE.type == "cuda":
            mlflow.log_metric("gpu_memory_peak_mb", torch.cuda.max_memory_allocated() / 1e6)
        log_figure(plot_confusion_matrix(data["y_test_np"], test_proba, f"{model_name} — test (segments)"),
                   "confusion_matrix_test.png")
        log_figure(plot_history(best.history, best.best_epoch, f"{model_name} — meilleure combinaison"),
                   "learning_curves.png")

        model_cpu = model.to("cpu").eval()
        example = data["X_test"][:2].cpu()
        signature = infer_signature(example.numpy(), model_cpu(example).detach().numpy())
        mlflow.pytorch.log_model(model_cpu, artifact_path="model", signature=signature)

    return {"model": model_name, **{f"val_{k}": v for k, v in best.val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()}}