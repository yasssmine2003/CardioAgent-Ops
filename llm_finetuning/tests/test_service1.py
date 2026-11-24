"""Tests du Service 1 (llm-server).

1. Dataset final : intégrité, séparation train / test, cohérence des règles Service 3.
2. Modelfile : même consigne et mêmes balises qu'à l'entraînement.
3. Schéma Pydantic de la réponse (garde-fou).
4. Intégration : appel réel à Ollama (sauté si llm-server n'est pas démarré).
"""
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.build_dataset import REQUIRED_KEYS, SYSTEM_PROMPT
from scripts.make_service3_examples import BORDERLINE
from src.ollama_client import chat, is_available
from src.schemas import InterpretationLLM, parse_llm_answer

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final"
MODELFILE = ROOT / "ollama" / "Modelfile"


def load(name: str) -> list[dict]:
    path = FINAL / name
    if not path.exists():
        pytest.skip(f"{path} absent (lancer scripts/build_dataset.py)")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def engine_output(conv: dict) -> dict:
    return json.loads(conv["messages"][1]["content"].split("Résultat du moteur audio :\n", 1)[1])


# ============================ 1. Dataset ============================

def test_conversations_are_well_formed():
    for conv in load("train.jsonl") + load("test.jsonl"):
        assert [m["role"] for m in conv["messages"]] == ["system", "user", "assistant"]
        assert conv["messages"][0]["content"] == SYSTEM_PROMPT
        assert all(m["content"].strip() for m in conv["messages"])


def test_no_test_question_in_train():
    train_q = {c["messages"][1]["content"] for c in load("train.jsonl")}
    assert not any(c["messages"][1]["content"] in train_q for c in load("test.jsonl"))


def test_service3_answers_follow_the_rules():
    for conv in load("train.jsonl") + load("test.jsonl"):
        if conv["family"] != "service3":
            continue
        x, answer = engine_output(conv), json.loads(conv["messages"][2]["content"])
        assert REQUIRED_KEYS == answer.keys()
        p, thr = x["probability"], x["model"]["threshold"]
        assert answer["resultat"] == ("pathologique" if p >= thr else "normal")
        assert answer["cas_limite"] == (abs(p - thr) < BORDERLINE)


# ============================ 2. Modelfile ============================

def test_modelfile_matches_training():
    if not MODELFILE.exists():
        pytest.skip("Modelfile absent")
    text = MODELFILE.read_text(encoding="utf-8")
    system = re.search(r'SYSTEM """(.*?)"""', text, re.S).group(1)
    assert system == SYSTEM_PROMPT                                   # même consigne qu'à l'entraînement
    assert re.search(r"^FROM \./.+\.gguf$", text, re.M)
    assert re.search(r"^PARAMETER temperature 0$", text, re.M)
    assert '<|im_start|>' in text and 'PARAMETER stop "<|im_end|>"' in text
    for line in re.findall(r"^PARAMETER .*$", text, re.M):           # pas de texte parasite après la valeur
        assert len(line.split(maxsplit=2)) == 3 and "→" not in line


# ============================ 3. Schéma Pydantic ============================

VALID = {"resultat": "normal", "fiabilite": "faible", "cas_limite": True,
         "interpretation": "Probabilité très proche du seuil.", "recommandation": "Refaire un enregistrement.",
         "avertissement": "Outil d'aide au dépistage."}


def test_schema_accepts_valid_answer_with_text_around():
    parsed = parse_llm_answer("Voici :\n" + json.dumps(VALID, ensure_ascii=False))
    assert parsed.cas_limite is True


@pytest.mark.parametrize("change", [{"resultat": "malade"}, {"fiabilite": "haute"}, {"diagnostic": "sténose"}])
def test_schema_rejects_invented_values_or_fields(change):
    with pytest.raises(ValidationError):
        InterpretationLLM.model_validate({**VALID, **change})


def test_schema_rejects_missing_field():
    with pytest.raises(ValidationError):
        InterpretationLLM.model_validate({k: v for k, v in VALID.items() if k != "recommandation"})


# ============================ 4. Intégration Ollama ============================

@pytest.mark.skipif(not is_available(), reason="llm-server (Ollama) non démarré")
def test_llm_server_answers_with_valid_schema():
    conv = next(c for c in load("test.jsonl") if c["family"] == "service3")
    answer, stats = chat(conv["messages"][1]["content"])
    parsed = parse_llm_answer(answer)
    assert parsed.resultat == json.loads(conv["messages"][2]["content"])["resultat"]
    assert stats["output_tokens"] > 0
