"""Étape 1.3 — Génère les exemples « sortie du Service 3 -> interprétation JSON », en français.

Usage (depuis llm_finetuning/) :  python -m scripts.make_service3_examples

Sortie : data/clean/service3.jsonl  (format : source, question, answer)
- question = consigne + sortie du moteur audio (JSON)
- answer   = interprétation au format JSON fixe (resultat, fiabilite, cas_limite,
             interpretation, recommandation, avertissement)

Les réponses sont écrites par des RÈGLES fixées par nous [C] (voir decide()), pas inventées :
le modèle apprendra ces règles en voyant les exemples.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "clean" / "service3.jsonl"
SEED = 42
N_EXAMPLES = 400

# ---------- Règles [C] ----------
DEPLOYED_THRESHOLD = 0.6557    # seuil du CNN déployé (Service 3)
BORDERLINE = 0.05              # écart au seuil < 0,05  -> cas limite, fiabilité faible
CLEAR = 0.20                   # écart >= 0,20          -> fiabilité élevée (entre les deux : moyenne)
SEGMENT_S = 5.0
LEVELS = ["faible", "moyenne", "élevée"]
WARNING = "Outil d'aide au dépistage : il ne remplace pas un examen médical."

# ---------- Formulations (plusieurs tournures pour éviter la récitation) ----------
INSTRUCTIONS = [
    "Interprète le résultat suivant du moteur audio CardioAgent et réponds uniquement avec le JSON demandé.",
    "Voici la sortie du moteur d'analyse des sons cardiaques. Donne ton interprétation au format JSON.",
    "Analyse ce résultat du moteur audio (enregistrement phonocardiographique) et réponds en JSON.",
]
BORDERLINE_TXT = [
    "très proche du seuil de décision ({thr}) : le modèle hésite entre normal et pathologique.",
    "presque égale au seuil de décision ({thr}) : le résultat est incertain.",
    "à seulement {gap} du seuil de décision ({thr}) : il s'agit d'un cas limite.",
]
MEDIUM_TXT = [
    "{side} du seuil de décision ({thr}), avec un écart modéré de {gap}.",
    "{side} du seuil ({thr}) ; l'écart de {gap} donne un résultat assez net.",
]
CLEAR_TXT = [
    "nettement {side} du seuil de décision ({thr}).",
    "{side} du seuil ({thr}) avec un écart important de {gap} : le résultat est net.",
]
PADDED_TXT = [
    "L'enregistrement est court ({dur} s) et a été complété par répétition : il contient moins d'information.",
    "Enregistrement court ({dur} s), complété par répétition jusqu'à 5 s : la fiabilité est réduite.",
]
DISAGREE_TXT = [
    "Les segments ne sont pas d'accord entre eux ({k} sur {n} au-dessus du seuil) : le signal est variable.",
    "Résultats hétérogènes selon les segments ({k} sur {n} au-dessus du seuil).",
]
SEGMENTS_TXT = [
    "Analyse réalisée sur {n} segments de 5 s.",
    "{n} segments de 5 s ont été analysés.",
]
RECO = {
    "limite": [
        "Refaire un enregistrement dans un endroit calme et demander l'avis d'un médecin.",
        "Résultat à confirmer : refaire l'enregistrement et faire vérifier par un médecin.",
    ],
    "pathologique": [
        "Consulter un cardiologue pour un examen clinique complet.",
        "Un avis cardiologique est recommandé pour confirmer ou écarter une anomalie.",
    ],
    "normal_net": [
        "Aucune anomalie détectée par l'outil. En cas de symptômes (essoufflement, douleur thoracique, palpitations), consulter un médecin.",
        "Pas d'anomalie détectée. Consulter un médecin si des symptômes apparaissent (essoufflement, douleur thoracique, palpitations).",
    ],
    "normal_moyen": [
        "Résultat plutôt normal ; en parler lors d'une prochaine consultation, surtout en cas de symptômes.",
        "Résultat probablement normal ; un contrôle médical est conseillé en cas de symptômes.",
    ],
    "court": [
        " Un enregistrement plus long (au moins 10 s) améliorerait la fiabilité.",
        " Refaire un enregistrement plus long (10 s ou plus) si possible.",
    ],
}


def fr(x: float, nd: int = 2) -> str:
    """Nombre au format français : 0,65."""
    return f"{x:.{nd}f}".replace(".", ",")


# ============================ 1. Entrées réalistes ============================

def sample_input(rng: random.Random) -> dict:
    """Une sortie plausible du Service 3 (seuls les champs utiles au LLM)."""
    thr = DEPLOYED_THRESHOLD if rng.random() < 0.7 else round(rng.uniform(0.40, 0.75), 4)
    padded = rng.random() < 0.15
    duration = round(rng.uniform(2.0, 4.9), 2) if padded else round(rng.uniform(5.0, 60.0), 2)
    n = 1 if padded else int(duration // SEGMENT_S)

    kind = rng.choices(["limite", "moyen", "net"], weights=[0.25, 0.30, 0.45])[0]
    gap = {"limite": rng.uniform(0.0, BORDERLINE), "moyen": rng.uniform(BORDERLINE, CLEAR),
           "net": rng.uniform(CLEAR, 0.6)}[kind]
    center = min(max(thr + rng.choice([-1, 1]) * gap, 0.01), 0.99)

    spread = 0.25 if (n >= 3 and rng.random() < 0.2) else 0.04      # 20 % : segments en désaccord
    segs = [round(min(max(rng.gauss(center, spread), 0.001), 0.999), 4) for _ in range(n)]
    p = round(sum(segs) / n, 4)                                     # comme le Service 3 : moyenne
    pathological = p >= thr
    return {"prediction": "pathologique" if pathological else "normal", "probability": p,
            "confidence": round(p if pathological else 1 - p, 4), "segment_probabilities": segs,
            "n_segments": n, "duration_s": duration, "padded": padded, "model": {"threshold": thr}}


# ============================ 2. Règles -> réponse ============================

def decide(x: dict, rng: random.Random) -> dict:
    p, thr, n = x["probability"], x["model"]["threshold"], x["n_segments"]
    gap = abs(p - thr)
    pathological = x["prediction"] == "pathologique"
    borderline = gap < BORDERLINE
    level = 0 if borderline else (1 if gap < CLEAR else 2)
    if x["padded"]:
        level = max(level - 1, 0)                                   # audio court : un niveau de moins
    above = sum(s >= thr for s in x["segment_probabilities"])
    disagree = n >= 2 and 0 < above < n

    side = "au-dessus" if pathological else "en dessous"
    template = rng.choice(BORDERLINE_TXT if borderline else MEDIUM_TXT if gap < CLEAR else CLEAR_TXT)
    parts = [f"Probabilité « pathologique » de {fr(p, 3)}, "
             + template.format(thr=fr(thr, 3), gap=fr(gap), side=side)]
    if x["padded"]:
        parts.append(rng.choice(PADDED_TXT).format(dur=fr(x["duration_s"], 1)))
    else:
        parts.append(rng.choice(SEGMENTS_TXT).format(n=n))
    if disagree:
        parts.append(rng.choice(DISAGREE_TXT).format(k=above, n=n))

    if borderline:
        reco = rng.choice(RECO["limite"])
    elif pathological:
        reco = rng.choice(RECO["pathologique"])
    else:
        reco = rng.choice(RECO["normal_net" if level == 2 else "normal_moyen"])
    if x["padded"]:
        reco += rng.choice(RECO["court"])

    return {"resultat": x["prediction"], "fiabilite": LEVELS[level], "cas_limite": borderline,
            "interpretation": " ".join(parts), "recommandation": reco, "avertissement": WARNING}


# ============================ 3. Écriture ============================

def make_example(rng: random.Random) -> dict:
    x = sample_input(rng)
    question = f"{rng.choice(INSTRUCTIONS)}\n\nRésultat du moteur audio :\n{json.dumps(x, ensure_ascii=False)}"
    return {"source": "service3", "question": question,
            "answer": json.dumps(decide(x, rng), ensure_ascii=False, indent=2)}


def main() -> None:
    rng = random.Random(SEED)
    examples = [make_example(rng) for _ in range(N_EXAMPLES)]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    answers = [json.loads(e["answer"]) for e in examples]
    count = lambda key, val: sum(a[key] == val for a in answers)
    print(f"{len(examples)} exemples -> {OUT_PATH}")
    print(f"  résultat     : normal {count('resultat', 'normal')} | pathologique {count('resultat', 'pathologique')}")
    print(f"  fiabilité    : " + " | ".join(f"{lv} {count('fiabilite', lv)}" for lv in LEVELS))
    print(f"  cas limites  : {count('cas_limite', True)}")
    print(f"  audios courts: {sum('complété par répétition' in a['interpretation'] for a in answers)}")
    print(f"  désaccords   : {sum('segments' in a['interpretation'] and 'au-dessus du seuil)' in a['interpretation'] for a in answers)}")
    for e in rng.sample(examples, 2):
        print(f"\n--- Exemple ---\n{e['question']}\n\n{e['answer']}")


if __name__ == "__main__":
    main()