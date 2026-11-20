"""Évaluation du LLM sur les 109 conversations de test (étape 2 : AVANT, étape 4 : APRÈS fine-tuning).

Trois mesures, une par type de question :
- service3 : le JSON est-il valide, complet, et les décisions (résultat, cas limite, fiabilité) justes ?
- medmcqa  : la bonne lettre (A, B, C, D) est-elle choisie ?            -> exactitude
- medquad  : la réponse ressemble-t-elle à la réponse de référence ?    -> ROUGE-L

Les fonctions de score n'ont pas besoin de GPU (testées sur PC) ; seule generate_answers() en a besoin.
"""
from __future__ import annotations

import json
import re
import time
from statistics import mean

REQUIRED_KEYS = {"resultat", "fiabilite", "cas_limite", "interpretation", "recommandation", "avertissement"}
MAX_NEW_TOKENS = {"service3": 300, "medmcqa": 300, "medquad": 400}   # [C] longueur maximale générée


# ============================ 1. Génération ============================

def generate_answers(model, tokenizer, conversations: list[dict], log_every: int = 10) -> list[dict]:
    """Fait répondre le modèle à chaque conversation (system + user), sans la réponse attendue.

    Décodage glouton (pas de hasard) : le même modèle donne toujours la même réponse,
    donc la comparaison avant / après est équitable.
    """
    import torch

    results = []
    for i, conv in enumerate(conversations, start=1):
        prompt = conv["messages"][:2]                                   # system + user seulement
        # return_dict=True : les versions récentes de transformers renvoient un dictionnaire
        # (input_ids + attention_mask), pas directement un tableau
        inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_prompt = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS[conv["family"]],
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, n_prompt:], skip_special_tokens=True).strip()
        results.append({"family": conv["family"], "source": conv["source"],
                        "question": conv["messages"][1]["content"],
                        "reference": conv["messages"][2]["content"], "prediction": answer,
                        "seconds": round(time.perf_counter() - start, 2),
                        "new_tokens": int(output.shape[1] - n_prompt)})
        if i % log_every == 0:
            print(f"{i}/{len(conversations)} réponses générées")
    return results


# ============================ 2. Scores ============================

def extract_json(text: str) -> dict | None:
    """Le premier objet JSON du texte (le modèle peut ajouter du texte autour)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_service3(prediction: str, reference: str) -> dict:
    ref, pred = json.loads(reference), extract_json(prediction)
    valid = pred is not None
    complete = valid and REQUIRED_KEYS <= pred.keys()
    same = lambda key: bool(valid and pred.get(key) == ref[key])
    return {"json_valide": valid, "json_complet": complete, "resultat_juste": same("resultat"),
            "cas_limite_juste": same("cas_limite"), "fiabilite_juste": same("fiabilite")}


LETTER_RE = [re.compile(r"Answer\s*[:\-]?\s*\(?([A-D])\b"),           # « Answer: C »
             re.compile(r"(?:réponse|answer)[^A-D]{0,20}\(?([A-D])\b", re.IGNORECASE),
             re.compile(r"^\s*\(?([A-D])[\.\):]", re.MULTILINE)]         # « C. Left Atrium »


def extract_letter(text: str) -> str | None:
    for pattern in LETTER_RE:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def score_medmcqa(prediction: str, reference: str) -> dict:
    return {"bonne_lettre": extract_letter(prediction) == extract_letter(reference)}


def score_medquad(prediction: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {"rougeL": round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)}


SCORERS = {"service3": score_service3, "medmcqa": score_medmcqa, "medquad": score_medquad}


def score_all(results: list[dict]) -> tuple[list[dict], dict]:
    """Ajoute les scores à chaque résultat et calcule les moyennes par type de question."""
    for r in results:
        r["scores"] = SCORERS[r["family"]](r["prediction"], r["reference"])
    summary = {}
    for family in SCORERS:
        rows = [r["scores"] for r in results if r["family"] == family]
        if rows:
            for key in rows[0]:
                summary[f"{family}_{key}"] = round(mean(float(s[key]) for s in rows), 4)
            summary[f"{family}_n"] = len(rows)
    summary["secondes_par_reponse"] = round(mean(r["seconds"] for r in results), 2)
    return results, summary