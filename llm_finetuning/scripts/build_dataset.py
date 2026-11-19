"""Étape 1.4 — Assemble le dataset final au format conversation et le découpe en train / test.

Usage (depuis llm_finetuning/) :  python -m scripts.build_dataset

Entrée  : data/clean/medquad.jsonl, medmcqa.jsonl, service3.jsonl   (étapes 1.2 et 1.3)
Sortie  : data/final/train.jsonl, data/final/test.jsonl, data/final/dataset_info.json

Une ligne = une conversation : {"source": ..., "messages": [system, user, assistant]}
C'est le format « messages » que TRL / Unsloth transforment avec le chat template de Qwen.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR, FINAL_DIR = ROOT / "data" / "clean", ROOT / "data" / "final"
SOURCES = {"medquad": "medquad.jsonl", "medmcqa": "medmcqa.jsonl", "service3": "service3.jsonl"}
SEED = 42
TEST_RATIO = 0.10                  # [C] 10 % mis de côté pour comparer avant / après fine-tuning
REQUIRED_KEYS = {"resultat", "fiabilite", "cas_limite", "interpretation", "recommandation", "avertissement"}

# [C] Consigne commune à toutes les conversations (identique à l'entraînement ET à l'inférence)
SYSTEM_PROMPT = (
    "Tu es l'assistant médical de la plateforme CardioAgent, spécialisé en cardiologie et en "
    "auscultation cardiaque. Tu réponds de façon exacte et prudente, sans inventer d'information. "
    "Quand on te donne un résultat du moteur audio, tu réponds uniquement avec le JSON demandé. "
    "Sinon, tu réponds dans la langue de la question. Tu ne poses pas de diagnostic définitif "
    "et tu ne prescris pas de traitement."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_conversation(example: dict, family: str) -> dict:
    return {"source": example["source"], "family": family, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]}


def check(conv: dict) -> None:
    """Contrôles : pas de texte vide ; les réponses « service3 » sont un JSON complet."""
    for m in conv["messages"]:
        if not m["content"].strip():
            raise ValueError(f"Message vide ({conv['source']})")
    if conv["family"] == "service3":
        answer = json.loads(conv["messages"][2]["content"])
        missing = REQUIRED_KEYS - answer.keys()
        if missing:
            raise ValueError(f"Champs manquants dans une réponse service3 : {missing}")


def split(convs: list[dict], rng: random.Random) -> tuple[list, list]:
    """10 % de chaque source dans le test (stratifié), le reste dans le train."""
    train, test = [], []
    for family in SOURCES:
        group = [c for c in convs if c["family"] == family]
        rng.shuffle(group)
        n_test = round(len(group) * TEST_RATIO)
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def n_chars(conv: dict) -> int:
    return sum(len(m["content"]) for m in conv["messages"])


def main() -> None:
    rng = random.Random(SEED)
    convs = []
    for family, filename in SOURCES.items():
        for example in load_jsonl(CLEAN_DIR / filename):
            conv = to_conversation(example, family)
            check(conv)
            convs.append(conv)

    train, test = split(convs, rng)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, FINAL_DIR / "train.jsonl")
    save_jsonl(test, FINAL_DIR / "test.jsonl")

    lengths = sorted(n_chars(c) for c in convs)
    info = {
        "seed": SEED, "test_ratio": TEST_RATIO, "system_prompt": SYSTEM_PROMPT,
        "train": dict(Counter(c["family"] for c in train)), "test": dict(Counter(c["family"] for c in test)),
        "chars_per_conversation": {"median": lengths[len(lengths) // 2], "max": lengths[-1]},
    }
    (FINAL_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Total : {len(convs)} conversations -> train {len(train)} | test {len(test)}")
    print(f"{'source':<10}{'train':>7}{'test':>6}")
    for family in SOURCES:
        print(f"{family:<10}{info['train'].get(family, 0):>7}{info['test'].get(family, 0):>6}")
    print(f"Longueur d'une conversation (caractères) : médiane {info['chars_per_conversation']['median']}, "
          f"max {info['chars_per_conversation']['max']}  (environ 4 caractères par token en anglais)")
    print(f"\n--- Exemple de conversation ({train[0]['family']}) ---")
    for m in train[0]["messages"]:
        print(f"[{m['role']}] {m['content'][:300]}")
    print(f"\nFichiers : {FINAL_DIR}")


if __name__ == "__main__":
    main()