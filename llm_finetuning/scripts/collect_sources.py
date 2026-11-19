"""Étape 1.1 — Télécharge MedQuAD et MedMCQA, garde la cardiologie, montre ce qu'on obtient.

Usage (depuis llm_finetuning/) :  python -m scripts.collect_sources

Produit dans data/sources/ (format commun : source, question, answer) :
- medquad_cardio.jsonl   questions-réponses NIH (patients)
- medmcqa_cardio.jsonl   QCM d'examen transformés en question -> réponse expliquée
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sources"
SEED = 42

# [C] Mots-clés « cardiologie » (anglais : les deux datasets sont en anglais)
CARDIO = re.compile(
    r"\b(heart|cardiac|cardio\w*|murmur\w*|valv\w*|mitral|aortic|tricuspid|myocard\w*|coronary|"
    r"angina|atri(al|um)|ventric\w*|arrhythmi\w*|tachycardi\w*|bradycardi\w*|fibrillation|"
    r"endocardit\w*|pericard\w*|heart sounds?|gallop|systolic|diastolic|ejection|"
    r"electrocardiogra\w*|ecg|echocardiogra\w*|hypertension)\b",
    re.IGNORECASE,
)
MIN_ANSWER_CHARS = 80   # [C] on écarte les réponses trop courtes pour apprendre quelque chose


def is_cardio(*texts: str) -> bool:
    return any(isinstance(t, str) and CARDIO.search(t) for t in texts)


# ============================== MedQuAD ==============================

def filter_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les réponses présentes, liées au cœur ; tout le dossier NHLBI n'est pas cardiaque (poumons, sang)."""
    df = df[df["answer"].fillna("").str.len() >= MIN_ANSWER_CHARS]
    keep = df.apply(lambda r: is_cardio(r["question"], r["question_focus"]), axis=1)
    df = df[keep].drop_duplicates(subset=["question", "answer"])
    return pd.DataFrame({"source": "medquad/" + df["document_source"].astype(str),
                         "question": df["question"].str.strip(),
                         "answer": df["answer"].str.strip()})


# ============================== MedMCQA ==============================
LETTERS = "ABCD"


def mcq_to_qa(row: pd.Series) -> dict:
    """QCM -> question avec options, et réponse = bonne option + explication."""
    options = [row["opa"], row["opb"], row["opc"], row["opd"]]
    question = row["question"].strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, options))
    k = int(row["cop"])
    answer = f"Answer: {LETTERS[k]}. {options[k]}\nExplanation: {row['exp'].strip()}"
    return {"source": f"medmcqa/{row['subject_name']}", "question": question, "answer": answer}


def filter_medmcqa(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les QCM à réponse unique, avec une explication utile, sur le cœur."""
    df = df[(df["choice_type"] == "single") & (df["exp"].fillna("").str.len() >= MIN_ANSWER_CHARS)]
    keep = df.apply(lambda r: is_cardio(r["question"], r["topic_name"]), axis=1)
    df = df[keep].drop_duplicates(subset=["question"])
    return pd.DataFrame([mcq_to_qa(r) for _, r in df.iterrows()])


# ============================== Bilan ==============================

def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def report(name: str, raw: pd.DataFrame, kept: pd.DataFrame, by: str) -> None:
    print(f"\n===== {name} : {len(raw)} lignes -> {len(kept)} gardées (cardiologie) =====")
    print("Par source / matière :", dict(collections.Counter(kept["source"]).most_common(8)))
    print("Longueur des réponses (caractères) :", kept["answer"].str.len().describe()[["min", "50%", "max"]].round().to_dict())
    print(f"Toutes les valeurs de « {by} » avant filtrage :", dict(collections.Counter(raw[by]).most_common(25)))
    for _, r in kept.sample(min(2, len(kept)), random_state=SEED).iterrows():
        print(f"\n--- Exemple ({r['source']}) ---\nQ : {r['question'][:400]}\nR : {r['answer'][:500]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    medquad = load_dataset("lavita/MedQuAD", split="train").to_pandas()
    mq = filter_medquad(medquad)
    save_jsonl(mq, OUT_DIR / "medquad_cardio.jsonl")
    report("MedQuAD", medquad, mq, "document_source")

    medmcqa = pd.concat([load_dataset("openlifescienceai/medmcqa", split=s).to_pandas()
                         for s in ("train", "validation")])   # le split test n'a pas les réponses
    mc = filter_medmcqa(medmcqa)
    save_jsonl(mc, OUT_DIR / "medmcqa_cardio.jsonl")
    report("MedMCQA", medmcqa, mc, "subject_name")

    print(f"\nFichiers écrits dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
