"""Exporte le modèle sélectionné en ONNX, vérifie l'export et fixe le seuil de déploiement.

Usage (depuis audio_engine/) :  python -m scripts.export_onnx

Produit dans models/ (ou AUDIO_ENGINE_MODELS_DIR) :
- cardio_model.onnx  le modèle (livrable imposé par le cahier des charges)
- cardio_model.json  la fiche : seuil, réglages du prétraitement, résultats
et un run MLflow « onnx_export » contenant les deux fichiers.
"""
import logging

import mlflow

from src import config
from src.export_onnx import export_best_model
from src.mlflow_tracking import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logging.getLogger("mlflow").setLevel(logging.ERROR)


def main() -> None:
    setup_mlflow()
    card = export_best_model()

    with mlflow.start_run(run_name="onnx_export"):
        mlflow.set_tags({"family": "deployment", "exported_model": card["model_name"],
                         "source_run_id": card["mlflow_run_id"]})
        mlflow.log_params({"deploy_min_specificity": config.DEPLOY_MIN_SPECIFICITY,
                           "deploy_threshold": round(card["decision"]["threshold"], 4),
                           "onnx_opset": config.ONNX_OPSET})
        for scope, values in card["metrics"].items():
            if isinstance(values, dict):
                mlflow.log_metrics({f"{scope}_{k}": v for k, v in values.items()})
        mlflow.log_metric("onnx_max_abs_diff", card["metrics"]["onnx_max_abs_diff"])
        mlflow.log_artifact(str(config.ONNX_MODEL_PATH))
        mlflow.log_artifact(str(config.MODEL_CARD_PATH))

    rec, seg = card["metrics"]["test_recording"], card["metrics"]["test_segment"]
    print(f"\n=== Modèle déployé : {card['model_name']} ===")
    print(f"Seuil (par enregistrement)  : {card['decision']['threshold']:.4f}")
    print(f"Écart ONNX / modèle d'origine : {card['metrics']['onnx_max_abs_diff']:.2e}")
    print(f"Test par enregistrement : Recall patho {rec['recall_pathological']:.1%} | "
          f"spécificité {rec['specificity']:.1%} | accuracy {rec['accuracy']:.1%} | AUC {rec['roc_auc']:.3f}")
    print(f"Test par segment        : Recall patho {seg['recall_pathological']:.1%} | "
          f"spécificité {seg['specificity']:.1%} | AUC {seg['roc_auc']:.3f}")
    print(f"\nFichiers : {config.ONNX_MODEL_PATH}\n           {config.MODEL_CARD_PATH}")


if __name__ == "__main__":
    main()