"""Étape 1.2 — Nettoie et échantillonne les exemples publics (MedQuAD, MedMCQA).

Usage (depuis llm_finetuning/) :  python -m scripts.clean_sample

Entrée  : data/sources/medquad_cardio.jsonl, data/sources/medmcqa_cardio.jsonl  (étape 1.1)
Sortie  : data/clean/medquad.jsonl, data/clean/medmcqa.jsonl  (format : source, question, answer)

Les 5 étapes (chacune affiche combien d'exemples il reste) :
  1. retirer les sources / matières hors sujet
  2. nettoyer le texte
  3. limiter la longueur des réponses
  4. retirer les doublons
  5. échantillonner (tirage reproductible, graine 42)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from spellchecker import SpellChecker

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR, OUT_DIR = ROOT / "data" / "sources", ROOT / "data" / "clean"
SEED = 42

# ---------- Réglages [C] ----------
MEDQUAD_DROP = {"medquad/GHR", "medquad/GARD"}            # maladies génétiques / rares : hors dépistage
MEDMCQA_KEEP = {"medmcqa/Medicine", "medmcqa/Physiology", "medmcqa/Pathology",
                "medmcqa/Anatomy", "medmcqa/Pediatrics", "medmcqa/Surgery"}   # matières proches de la clinique
MEDQUAD_MAX_CHARS = 1500     # réponse coupée proprement au-delà (environ 350 tokens)
MEDMCQA_CHARS = (150, 1200)  # explication gardée seulement dans cette plage (pas de coupe au milieu)
MIN_CHARS = 80
N_MEDQUAD, N_MEDMCQA = 350, 500


# ============================ 2. Nettoyage du texte ============================
# Apostrophes perdues dans MedQuAD (« cant » -> « can't »)
CONTRACTIONS = {"cant": "can't", "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't",
                "wont": "won't", "didnt": "didn't", "youre": "you're", "youll": "you'll", "youve": "you've",
                "theyre": "they're", "wouldnt": "wouldn't", "shouldnt": "shouldn't", "couldnt": "couldn't"}
CONTRACTION_RE = re.compile(r"\b(" + "|".join(CONTRACTIONS) + r")\b", re.IGNORECASE)
# Début d'explication MedMCQA : « Ans. is 'c' i.e., » / « Answer: (b) »
ANS_PREFIX_RE = re.compile(r"^\s*Ans(?:wer)?[\s.:\-]*(?:is)?\s*['\"(]?[a-dA-D]['\")]?\s*(?:i\.?\s*e\.?\s*,?)?\s*",
                           re.IGNORECASE)
REF_RE = re.compile(r"\(?\s*Ref(?:erence)?s?\s*[:.\-].*?(?:\)|$)", re.IGNORECASE | re.MULTILINE)


# Défaut d'origine de MedMCQA : les lettres « rt » ont disparu (« hea » = heart, « aery » = artery).
# Réparation : si le mot + « rt » inséré donne un mot anglais BEAUCOUP plus fréquent, on corrige.
WORD_FREQ = SpellChecker().word_frequency
RT_OVERRIDES = {"aerial": "arterial"}     # « aerial » existe en anglais, mais ici c'est toujours « arterial »
WORD_RE = re.compile(r"[A-Za-z]+")


def _freq(word: str) -> int:
    return WORD_FREQ[word] if word in WORD_FREQ else 0


def fix_missing_rt(word: str) -> str:
    low = word.lower()
    if len(low) < 3 or word.isupper():                     # sigles (PA, ECG) et mots courts : on n'y touche pas
        return word
    best = RT_OVERRIDES.get(low) or max((low[:i] + "rt" + low[i:] for i in range(1, len(low) + 1)), key=_freq)
    if low in RT_OVERRIDES or (_freq(best) >= 50 and _freq(best) > 20 * _freq(low)):
        return best.capitalize() if word[0].isupper() else best
    return word


def repair_rt(text: str) -> str:
    return WORD_RE.sub(lambda m: fix_missing_rt(m.group(0)), text)


def tidy(text: str) -> str:
    """Espaces multiples et lignes vides en trop."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)          # espaces en bout de ligne
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def clean_medquad(answer: str) -> str:
    def fix(m: re.Match) -> str:                        # garde la majuscule : « Dont » -> « Don't »
        word = CONTRACTIONS[m.group(0).lower()]
        return word.capitalize() if m.group(0)[0].isupper() else word
    return tidy(CONTRACTION_RE.sub(fix, answer))


def clean_medmcqa(answer: str) -> str:
    """Réparation des « rt » manquants, puis :
    « Answer: C. Left Atrium\\nExplanation: Ans. is 'c' i.e., Left AtriumCardiac myxoma... » -> explication propre."""
    head, _, explanation = repair_rt(answer).partition("\nExplanation: ")
    option_text = head.split(". ", 1)[-1].strip()                 # « Left Atrium »
    explanation = ANS_PREFIX_RE.sub("", explanation)                # retire « Ans. is 'c' i.e., »
    # retire la réponse répétée et collée (espaces ignorés : « 1/16 th » = « 1/16th »)
    repeat = r"\s*".join(re.escape(c) for c in option_text.replace(" ", ""))
    explanation = re.sub(rf"^\s*{repeat}[\s.,:;-]*", "", explanation, count=1, flags=re.IGNORECASE)
    explanation = REF_RE.sub("", explanation)                       # retire les références de manuels
    return f"{head}\nExplanation: {tidy(explanation)}"


# ============================ 3. Longueur ============================

def cut_clean(text: str, max_chars: int) -> str:
    """Coupe avant max_chars, à la fin d'un paragraphe ou à défaut d'une phrase (jamais au milieu)."""
    if len(text) <= max_chars:
        return text
    part = text[:max_chars]
    cut = part.rfind("\n\n")
    if cut < max_chars // 2:                       # pas de paragraphe assez loin : fin de phrase
        cut = max(part.rfind(". "), part.rfind(".\n")) + 1
    return part[:cut].strip() if cut > 0 else ""


def explanation_len(answer: str) -> int:
    return len(answer.partition("\nExplanation: ")[2])


# ============================ Pipeline ============================

def load(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def step(log: list, name: str, df: pd.DataFrame) -> pd.DataFrame:
    log.append((name, len(df)))
    return df


def process_medquad(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    log = [("départ", len(df))]
    df = step(log, "1. sources hors sujet retirées", df[~df["source"].isin(MEDQUAD_DROP)])
    df = df.assign(answer=df["answer"].map(clean_medquad))
    df = step(log, "2. texte nettoyé", df)
    df = df.assign(answer=df["answer"].map(lambda a: cut_clean(a, MEDQUAD_MAX_CHARS)))
    df = step(log, "3. réponses coupées <= 1500 car.", df[df["answer"].str.len() >= MIN_CHARS])
    df = step(log, "4. doublons retirés", df.drop_duplicates(subset=["question"]))
    df = step(log, "5. échantillon", df.sample(min(N_MEDQUAD, len(df)), random_state=SEED))
    return df, log


def process_medmcqa(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    log = [("départ", len(df))]
    df = step(log, "1. matières hors sujet retirées", df[df["source"].isin(MEDMCQA_KEEP)])
    df = df.assign(question=df["question"].map(repair_rt), answer=df["answer"].map(clean_medmcqa))
    df = step(log, "2. texte nettoyé", df)
    lo, hi = MEDMCQA_CHARS
    n = df["answer"].map(explanation_len)
    df = step(log, f"3. explications de {lo} à {hi} car.", df[(n >= lo) & (n <= hi)])
    df = step(log, "4. doublons retirés", df.drop_duplicates(subset=["question"]))
    # 5. échantillon proportionnel à chaque matière (garde l'équilibre d'origine)
    frac = min(1.0, N_MEDMCQA / len(df))
    df = df.groupby("source", group_keys=False).sample(frac=frac, random_state=SEED)
    df = step(log, "5. échantillon", df)
    return df, log


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in df[["source", "question", "answer"]].to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def show(name: str, df: pd.DataFrame, log: list) -> None:
    print(f"\n===== {name} =====")
    for label, n in log:
        print(f"  {label:<36} {n:>5}")
    print("  Par source :", df["source"].value_counts().to_dict())
    print("  Longueur des réponses (médiane / max) :", int(df["answer"].str.len().median()), "/",
          int(df["answer"].str.len().max()))
    for _, r in df.sample(min(2, len(df)), random_state=SEED).iterrows():
        print(f"\n  --- Exemple ({r['source']}) ---\n  Q : {r['question'][:300]}\n  R : {r['answer'][:600]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mq, mq_log = process_medquad(load(SRC_DIR / "medquad_cardio.jsonl"))
    mc, mc_log = process_medmcqa(load(SRC_DIR / "medmcqa_cardio.jsonl"))
    save_jsonl(mq, OUT_DIR / "medquad.jsonl")
    save_jsonl(mc, OUT_DIR / "medmcqa.jsonl")
    show("MedQuAD", mq, mq_log)
    show("MedMCQA", mc, mc_log)
    print(f"\nTotal public : {len(mq) + len(mc)} exemples -> {OUT_DIR}")


if __name__ == "__main__":
    main()