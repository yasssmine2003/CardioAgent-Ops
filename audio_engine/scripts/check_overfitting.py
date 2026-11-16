"""Y a-t-il du sur-apprentissage ? Deux mesures, sans ré-entraîner (tout est relu dans MLflow).

Usage (sur Colab, depuis audio_engine/) :  python -m scripts.check_overfitting

1. Courbes d'apprentissage des 3 réseaux : perte train / validation par époque (meilleure combinaison).
   Sur-apprentissage = la perte de validation remonte alors que la perte train continue de baisser.
2. AUC sur train, validation et test pour les 5 modèles.
   Sur-apprentissage = AUC train nettement plus haute que l'AUC validation / test.

Produit dans data/reports/overfitting/ (ou reports_cycles/) :
    learning_curves.png, auc_train_val_test.png, overfitting.csv
"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from src import config
from src.benchmark import find_latest_run, load_inputs, load_model, predict_proba
from src.mlflow_tracking import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger = logging.getLogger("check_overfitting")

DEEP = ["cnn", "resnet18", "transformer_cnn"]
ALL = ["decision_tree", "svm", *DEEP]
OUT_DIR = config.REPORTS_DIR / "overfitting"
# [C] Règle de lecture de l'écart d'AUC train - validation
GAP_OK, GAP_LIGHT = 0.03, 0.08


def verdict(gap: float) -> str:
    return "non" if gap < GAP_OK else ("léger" if gap < GAP_LIGHT else "oui")


# ============== 1. Courbes d'apprentissage (réseaux) ==============

def best_child_history(model_name: str) -> dict:
    """Historique par époque de la meilleure combinaison (run « enfant » au meilleur score)."""
    parent = find_latest_run(model_name)
    children = mlflow.search_runs(experiment_ids=[parent["experiment_id"]],
                                  filter_string=f"tags.mlflow.parentRunId = '{parent['run_id']}'",
                                  order_by=["metrics.selection_score DESC"])
    child = children.iloc[0]
    client = MlflowClient()
    history = {}
    for key in ("train_loss", "val_loss"):
        points = sorted(client.get_metric_history(child["run_id"], key), key=lambda m: m.step)
        history[key] = np.array([m.value for m in points])
    history["best_epoch"] = int(child["metrics.best_epoch"])
    return history


def plot_learning_curves(histories: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))
    for ax, (name, h) in zip(np.atleast_1d(axes), histories.items()):
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="train", color="tab:blue")
        ax.plot(epochs, h["val_loss"], label="validation", color="tab:orange")
        ax.axvline(h["best_epoch"], ls=":", color="black", lw=1, label=f"époque gardée ({h['best_epoch']})")
        ax.set(title=name, xlabel="Époque", ylabel="Perte (BCE pondérée)")
        ax.legend()
    fig.suptitle(f"Courbes d'apprentissage — mode {getattr(config, 'FEATURE_SET', 'fixed')}")
    fig.tight_layout()
    return fig


def curve_summary(h: dict) -> dict:
    """Chiffres lus sur la courbe : la perte de validation remonte-t-elle après son minimum ?"""
    best = h["best_epoch"] - 1
    min_epoch = int(np.argmin(h["val_loss"]))
    return {"epoque_gardee": h["best_epoch"],
            "train_loss_gardee": round(float(h["train_loss"][best]), 3),
            "val_loss_gardee": round(float(h["val_loss"][best]), 3),
            "val_loss_min_epoque": min_epoch + 1,
            "val_loss_remontee_fin": round(float(h["val_loss"][-1] - h["val_loss"][min_epoch]), 3)}


# ============== 2. AUC train / validation / test (5 modèles) ==============

def auc_by_split(model_name: str) -> dict:
    loaded = load_model(model_name)
    row = {"model": model_name}
    for split in ("train", "val", "test"):
        X, y, _ = load_inputs(loaded.family, split)
        row[f"auc_{split}"] = round(float(roc_auc_score(y, predict_proba(loaded, X))), 3)
    row["ecart_train_val"] = round(row["auc_train"] - row["auc_val"], 3)
    row["sur_apprentissage"] = verdict(row["ecart_train_val"])
    return row


def plot_auc(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    x, width = np.arange(len(table)), 0.27
    for k, (split, color) in enumerate([("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]):
        bars = ax.bar(x + (k - 1) * width, table[f"auc_{split}"], width, label=split, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, table["model"])
    ax.set(ylim=(0.5, 1.05), ylabel="AUC", title="AUC train / validation / test (écart = signe de sur-apprentissage)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_mlflow()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    histories = {}
    for name in DEEP:
        try:
            histories[name] = best_child_history(name)
        except LookupError as exc:
            logger.warning("%s", exc)
    if histories:
        plot_learning_curves(histories).savefig(OUT_DIR / "learning_curves.png", dpi=130)

    rows = []
    for name in ALL:
        logger.info("AUC train / val / test : %s", name)
        try:
            row = auc_by_split(name)
        except LookupError as exc:
            logger.warning("%s", exc)
            continue
        if name in histories:
            row.update(curve_summary(histories[name]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "overfitting.csv", index=False)
    plot_auc(table).savefig(OUT_DIR / "auc_train_val_test.png", dpi=130)

    pd.set_option("display.width", 250)
    print(f"\n=== Sur-apprentissage (écart AUC train - val : < {GAP_OK} non, < {GAP_LIGHT} léger, sinon oui) ===")
    print(table.to_string(index=False))
    print(f"\nFigures : {OUT_DIR / 'learning_curves.png'}\n          {OUT_DIR / 'auc_train_val_test.png'}")


if __name__ == "__main__":
    main()