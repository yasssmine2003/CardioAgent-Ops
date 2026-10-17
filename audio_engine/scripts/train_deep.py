"""Entraîne les modèles de deep learning avec suivi MLflow (GPU recommandé : Colab).

Usage (depuis audio_engine/) :
    python -m scripts.train_deep --models cnn
"""
import argparse
import logging

import pandas as pd

from src.mlflow_tracking import setup_mlflow
from src.models import DEEP_MODELS
from src.training.train_deep import run_deep

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logging.getLogger("mlflow").setLevel(logging.ERROR)

COLUMNS = ["model", "val_recall_pathological", "val_specificity", "val_f1",
           "test_recall_pathological", "test_specificity", "test_accuracy", "test_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEEP_MODELS), choices=list(DEEP_MODELS))
    args = parser.parse_args()

    setup_mlflow()
    results = [run_deep(name) for name in args.models]

    print("\n=== Résultats (meilleure combinaison de chaque modèle) ===")
    print(pd.DataFrame(results)[COLUMNS].round(3).to_string(index=False))


if __name__ == "__main__":
    main()