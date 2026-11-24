"""Petit client de l'API Ollama (llm-server) : une question -> réponse + mesures d'inférence."""
from __future__ import annotations

import json
import os
import time
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")   # dans Docker : http://llm-server:11434
MODEL = os.getenv("OLLAMA_MODEL", "cardio-llm")


def chat(question: str, url: str = OLLAMA_URL, model: str = MODEL, timeout: int = 600) -> tuple[str, dict]:
    """Envoie une question (sans message system : celui du Modelfile est utilisé).

    Retourne (réponse, mesures). Les durées d'Ollama sont en nanosecondes ; on les convertit en secondes.
    """
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": question}]}).encode()
    request = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    wall = time.perf_counter() - start
    ns = 1e9
    eval_s = data.get("eval_duration", 0) / ns
    stats = {
        "wall_s": round(wall, 2),                                       # temps vu par le client
        "load_s": round(data.get("load_duration", 0) / ns, 2),          # chargement du modèle en mémoire
        "prompt_tokens": data.get("prompt_eval_count", 0),              # tokens de la question lus
        "prompt_s": round(data.get("prompt_eval_duration", 0) / ns, 2),
        "output_tokens": data.get("eval_count", 0),                     # tokens écrits
        "output_s": round(eval_s, 2),
        "tokens_per_s": round(data.get("eval_count", 0) / eval_s, 2) if eval_s else 0.0,
    }
    return data["message"]["content"], stats


def is_available(url: str = OLLAMA_URL, model: str = MODEL) -> bool:
    """Vrai si Ollama répond et connaît le modèle (utilisé pour sauter les tests d'intégration)."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as response:
            names = [m["name"] for m in json.loads(response.read()).get("models", [])]
        return any(n.split(":")[0] == model for n in names)
    except OSError:
        return False
