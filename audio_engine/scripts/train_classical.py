"""Entraîne les modèles classiques avec suivi MLflow.

Usage (depuis audio_engine/) :
    python -m scripts.train_classical                     # les deux modèles
    python -m scripts.train_classical --models svm        # un seul modèle
"""
import argparse
import logging

import pandas as pd

from src.mlflow_tracking import setup_mlflow
from src.models import CLASSICAL_MODELS
from src.training.train_classical import run_classical

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("mlflow").setLevel(logging.WARNING)

COLUMNS = ["model", "val_recall_pathological", "val_specificity", "val_f1",
           "test_recall_pathological", "test_specificity", "test_accuracy", "test_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(CLASSICAL_MODELS),
                        choices=list(CLASSICAL_MODELS))
    args = parser.parse_args()

    setup_mlflow()
    results = [run_classical(name) for name in args.models]

    print("\n=== Résultats (meilleure combinaison de chaque modèle) ===")
    print(pd.DataFrame(results)[COLUMNS].round(3).to_string(index=False))


if __name__ == "__main__":
    main()