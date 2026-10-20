"""Benchmark final des 5 modèles et sélection du meilleur (à lancer sur Colab).

Usage (depuis audio_engine/) :  python -m scripts.benchmark

Produit :
- data/reports/benchmark.csv   le tableau complet
- data/reports/best_model.json le modèle sélectionné et son seuil (utilisé pour l'export ONNX)
- un run MLflow « benchmark » avec le tableau, les courbes ROC et la sélection
"""
import json
import logging

import mlflow
import pandas as pd

from src import config
from src.benchmark import benchmark_models, plot_roc_curves, select_best_model
from src.mlflow_tracking import log_figure, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logging.getLogger("mlflow").setLevel(logging.ERROR)

DISPLAY = {
    "model": "modèle", "threshold": "seuil", "val_recall_pathological": "val_recall_patho",
    "val_auc": "val_auc", "recall_pathological": "recall_patho", "specificity": "spécificité",
    "accuracy": "accuracy", "f1": "f1", "auc": "auc", "missed_pathological": "malades_ratés",
    "false_alarms": "fausses_alertes", "rec_recall_pathological": "recall_patho_enreg",
    f"recall_pathological_at_spec{int(config.ALT_SPECIFICITY * 100)}":
        f"recall_patho_spé{int(config.ALT_SPECIFICITY * 100)}",
    "training_time_s": "entraînement_s", "inference_time_ms_cpu": "inférence_ms_cpu",
    "model_size_mb": "taille_Mo",
    "gpu_inference_memory_mb": "mém_gpu_inf_Mo",
    "gpu_inference_ms_per_segment": "inférence_ms_gpu",
}


def main() -> None:
    setup_mlflow()
    results, curves = benchmark_models()
    best = select_best_model(results)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.BENCHMARK_CSV_PATH, index=False)
    config.BEST_MODEL_PATH.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    with mlflow.start_run(run_name="benchmark"):
        mlflow.set_tags({"family": "benchmark", "selected_model": best["model_name"]})
        mlflow.log_params({"min_specificity": config.MIN_SPECIFICITY,
                           "selected_model": best["model_name"],
                           "selected_threshold": round(best["threshold"], 4),
                           "selected_run_id": best["run_id"]})
        mlflow.log_metrics({f"selected_test_{k}": v for k, v in best["test_metrics"].items()})
        mlflow.log_artifact(str(config.BENCHMARK_CSV_PATH))
        mlflow.log_artifact(str(config.BEST_MODEL_PATH))
        log_figure(plot_roc_curves(curves, best["model_name"]), "roc_curves_test.png")

    table = results[list(DISPLAY)].rename(columns=DISPLAY)
    pd.set_option("display.width", 250)
    print(f"\n=== Benchmark (test) — seuil choisi sur la validation pour une spécificité >= "
          f"{config.MIN_SPECIFICITY:.0%} ===")
    print(table.round(3).to_string(index=False))
    print(f"\n>>> Modèle sélectionné : {best['model_name']} (seuil {best['threshold']:.3f})")
    print(f"    Règle : {best['selection_rule']}")
    print(f"    Fichier : {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()