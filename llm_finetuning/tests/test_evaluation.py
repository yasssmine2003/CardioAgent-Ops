"""Tests des fonctions de score (sans GPU, sans modèle)."""
import json

import pytest

from src.evaluation import extract_json, extract_letter, score_all, score_service3

REF = json.dumps({"resultat": "normal", "fiabilite": "faible", "cas_limite": True, "interpretation": "x",
                  "recommandation": "y", "avertissement": "z"})


def test_json_found_inside_text():
    assert extract_json('Voici : {"a": 1} merci') == {"a": 1}
    assert extract_json("pas de json") is None
    assert extract_json("{cassé") is None


def test_service3_perfect_and_wrong():
    assert all(score_service3(REF, REF).values())
    wrong = REF.replace('"normal"', '"pathologique"')
    s = score_service3(wrong, REF)
    assert s["json_valide"] and not s["resultat_juste"] and s["cas_limite_juste"]
    assert not any(score_service3("Le patient va bien.", REF).values())


@pytest.mark.parametrize("text, letter", [
    ("Answer: C. Left Atrium\nExplanation: ...", "C"),
    ("The correct answer is (B) because...", "B"),
    ("C. Left Atrium is the most common site.", "C"),
    ("I am not sure.", None),
])
def test_extract_letter(text, letter):
    assert extract_letter(text) == letter


def test_score_all_summary():
    pytest.importorskip("rouge_score")
    results = [{"family": "service3", "prediction": REF, "reference": REF, "seconds": 2.0},
               {"family": "medmcqa", "prediction": "Answer: A", "reference": "Answer: B. x", "seconds": 1.0},
               {"family": "medquad", "prediction": "heart failure means", "reference": "heart failure means weak heart",
                "seconds": 3.0}]
    _, summary = score_all(results)
    assert summary["service3_resultat_juste"] == 1.0
    assert summary["medmcqa_bonne_lettre"] == 0.0
    assert 0 < summary["medquad_rougeL"] < 1
    assert summary["secondes_par_reponse"] == 2.0
