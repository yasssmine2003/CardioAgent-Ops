"""Étape 7 — Mesure l'inférence de llm-server (Ollama, processeur) et l'enregistre dans MLflow.

Usage (depuis llm_finetuning/, conteneur llm-server démarré) :
    python -m scripts.benchmark_inference                 # 10 questions Service 3 + 5 questions médicales
    python -m scripts.benchmark_inference --n-service3 4 --n-medical 2

Mesures : temps par réponse, vitesse (tokens/s), chargement, taux de JSON valides (schéma Pydantic)
et de décisions justes sur les questions Service 3 du TEST (jamais vues à l'entraînement).
MLflow : variable MLFLOW_TRACKING_URI (par défaut : dossier local llm_finetuning/mlruns).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import mean, median

import mlflow

from src.ollama_client import MODEL, OLLAMA_URL, chat, is_available
from src.schemas import parse_llm_answer

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "final" / "test.jsonl"
OUT_PATH = ROOT / "data" / "final" / "inference_benchmark.jsonl"


def load_test(n_service3: int, n_medical: int) -> list[dict]:
    with TEST_PATH.open(encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    service3 = [c for c in test if c["family"] == "service3"][:n_service3]
    medical = [c for c in test if c["family"] == "medquad"][:n_medical]
    return service3 + medical


def check_service3(answer: str, reference: str) -> dict:
    """Format valide (Pydantic) + décisions identiques à la référence."""
    ref = json.loads(reference)
    try:
        pred = parse_llm_answer(answer)
    except (ValueError, json.JSONDecodeError):          # ValidationError hérite de ValueError
        return {"schema_ok": False, "resultat_ok": False, "cas_limite_ok": False, "fiabilite_ok": False}
    return {"schema_ok": True, "resultat_ok": pred.resultat == ref["resultat"],
            "cas_limite_ok": pred.cas_limite == ref["cas_limite"], "fiabilite_ok": pred.fiabilite == ref["fiabilite"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-service3", type=int, default=10)
    parser.add_argument("--n-medical", type=int, default=5)
    args = parser.parse_args()

    if not is_available():
        raise SystemExit(f"Ollama ou le modèle « {MODEL} » ne répond pas sur {OLLAMA_URL}")

    print("Échauffement (chargement du modèle en mémoire)...")
    _, warm = chat("Bonjour")
    rows = []
    for i, conv in enumerate(load_test(args.n_service3, args.n_medical), start=1):
        answer, stats = chat(conv["messages"][1]["content"])
        row = {"family": conv["family"], **stats, "answer": answer}
        if conv["family"] == "service3":
            row.update(check_service3(answer, conv["messages"][2]["content"]))
        rows.append(row)
        print(f"{i:>2} {conv['family']:<9} {stats['wall_s']:>6.1f} s  {stats['output_tokens']:>4} tokens  "
              f"{stats['tokens_per_s']:>5.1f} tokens/s")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    s3 = [r for r in rows if r["family"] == "service3"]
    metrics = {
        "load_s_first_call": warm["load_s"],
        "latency_s_median": median(r["wall_s"] for r in rows),
        "latency_s_max": max(r["wall_s"] for r in rows),
        "tokens_per_s_mean": round(mean(r["tokens_per_s"] for r in rows), 2),
        "output_tokens_mean": round(mean(r["output_tokens"] for r in rows), 1),
        "prompt_tokens_mean": round(mean(r["prompt_tokens"] for r in rows), 1),
    }
    if s3:
        for key in ("schema_ok", "resultat_ok", "cas_limite_ok", "fiabilite_ok"):
            metrics[f"service3_{key}"] = round(mean(float(r[key]) for r in s3), 3)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
    mlflow.set_experiment("cardioagent-llm")
    with mlflow.start_run(run_name="inference_ollama_cpu"):
        mlflow.set_tags({"stage": "inference", "family": "serving", "runtime": "ollama-docker-cpu"})
        mlflow.log_params({"model": MODEL, "url": OLLAMA_URL, "n_service3": len(s3),
                           "n_medical": len(rows) - len(s3), "cpu": platform.processor() or platform.machine(),
                           "quantization": "Q4_K_M", "temperature": 0})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(OUT_PATH))

    print("\n=== Mesures d'inférence (enregistrées dans MLflow, run inference_ollama_cpu) ===")
    for key, value in metrics.items():
        print(f"  {key:<24} {value}")


if __name__ == "__main__":
    main()
