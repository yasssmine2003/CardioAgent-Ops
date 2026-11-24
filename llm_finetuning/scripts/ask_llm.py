"""Étape 6.3 — Interroge le llm-server (Ollama) : un résultat du Service 3 + une question médicale.

Usage (depuis llm_finetuning/, conteneur llm-server démarré) :
    python -m scripts.ask_llm
    python -m scripts.ask_llm --url http://localhost:11434

Pas de message « system » envoyé : Ollama utilise celui du Modelfile (identique à l'entraînement).
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import urllib.error
import urllib.request

MODEL = "cardio-llm"
REQUIRED_KEYS = {"resultat", "fiabilite", "cas_limite", "interpretation", "recommandation", "avertissement"}

# Même forme de question qu'à l'entraînement (scripts/make_service3_examples.py)
SERVICE3_OUTPUT = {"prediction": "normal", "probability": 0.6526, "confidence": 0.3474,
                   "segment_probabilities": [0.7548, 0.5503], "n_segments": 2, "duration_s": 10.39,
                   "padded": False, "model": {"threshold": 0.6557}}
QUESTIONS = {
    "service3": "Interprète le résultat suivant du moteur audio CardioAgent et réponds uniquement avec le JSON demandé."
                f"\n\nRésultat du moteur audio :\n{json.dumps(SERVICE3_OUTPUT, ensure_ascii=False)}",
    "medical": "What is a heart murmur?",
}


def chat(url: str, question: str) -> tuple[str, dict]:
    body = json.dumps({"model": MODEL, "stream": False,
                       "messages": [{"role": "user", "content": question}]}).encode()
    request = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:   # sur processeur : jusqu'à quelques minutes
        data = json.loads(response.read())
    return data["message"]["content"], data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()

    for name, question in QUESTIONS.items():
        print(f"\n===== {name} =====\nQUESTION : {question[:200]}...\n")
        start = time.perf_counter()
        try:
            answer, data = chat(args.url, question)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                sys.exit(f"Le modèle « {MODEL} » n'existe pas encore dans Ollama (création en cours ?).\n"
                         "Vérifier avec :  docker compose exec llm-server ollama list")
            raise
        except urllib.error.URLError:
            sys.exit(f"Ollama ne répond pas sur {args.url} : le conteneur llm-server est-il démarré ?")
        seconds = time.perf_counter() - start
        print(f"RÉPONSE :\n{answer}\n")
        tokens = data.get("eval_count", 0)
        speed = tokens / (data.get("eval_duration", 1) / 1e9) if tokens else 0
        print(f"Temps total : {seconds:.1f} s | {tokens} tokens générés | {speed:.1f} tokens/s")
        if name == "service3":
            try:
                parsed = json.loads(answer[answer.find("{"):answer.rfind("}") + 1])
                missing = REQUIRED_KEYS - parsed.keys()
                print("JSON valide, 6 champs présents" if not missing else f"JSON incomplet, il manque : {missing}")
            except json.JSONDecodeError:
                print("Réponse non JSON")


if __name__ == "__main__":
    main()