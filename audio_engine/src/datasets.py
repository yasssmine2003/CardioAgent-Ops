"""Data wrangling : inventaire des audios et découpage train/val/test sans fuite."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (DATASET_DIR, LABELS, RANDOM_SEED, RAW_DIR,
                        SOURCE_BY_SAMPLE_RATE, TEST_RATIO, VAL_RATIO)

logger = logging.getLogger(__name__)

# PASCAL B : "112_1306243000964_A.wav" -> identifiant "112"
PASCAL_SESSION = re.compile(r"^(\d+)_\d+_")


# ======================= 1. INVENTAIRE =======================

def describe_audio(path: Path) -> dict:
    """Lit l'en-tête d'un .wav (sans charger le signal) et retourne ses propriétés."""
    try:
        info = sf.info(path)
        return {"sample_rate": info.samplerate,
                "n_channels": info.channels,
                "duration_s": round(info.frames / info.samplerate, 3),
                "valid": info.frames > 0}
    except Exception as exc:
        logger.warning("Fichier illisible : %s (%s)", path.name, exc)
        return {"sample_rate": None, "n_channels": None,
                "duration_s": None, "valid": False}


def make_group_id(filename: str, source: str) -> str:
    """Identifiant de groupe : les fichiers d'un même groupe restent dans le même ensemble.

    - PASCAL B : préfixe numérique commun (plusieurs enregistrements d'une même personne)
    - Autres   : le fichier forme son propre groupe (pas d'identifiant patient disponible)
    """
    match = PASCAL_SESSION.match(filename)
    key = match.group(1) if match else Path(filename).stem
    return f"{source}:{key}"


def build_inventory(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Une ligne par fichier .wav : chemin, label, source, propriétés audio, groupe."""
    rows = []
    for label_name, label in LABELS.items():
        folder = dataset_dir / label_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dossier introuvable : {folder}")
        for wav in sorted(folder.glob("*.wav")):
            props = describe_audio(wav)
            source = SOURCE_BY_SAMPLE_RATE.get(props["sample_rate"], "inconnue")
            rows.append({
                "path": wav.relative_to(RAW_DIR).as_posix(),
                "filename": wav.name,
                "label": label,
                "label_name": label_name,
                "source": source,
                "group_id": make_group_id(wav.name, source),
                **props,
            })
    df = pd.DataFrame(rows)
    logger.info("Inventaire : %d fichiers, dont %d invalides",
                len(df), int((~df["valid"]).sum()))
    return df


# ======================= 2. DÉCOUPAGE =======================

def _pick_one_fold(df: pd.DataFrame, n_parts: int, seed: int) -> pd.Index:
    """Coupe df en n_parts parts équilibrées et groupées, et retourne l'une d'elles."""
    strata = df["source"] + "_" + df["label"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_parts, shuffle=True, random_state=seed)
    _, fold_positions = next(splitter.split(df, strata, groups=df["group_id"]))
    return df.index[fold_positions]


def assign_splits(df: pd.DataFrame, val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Ajoute la colonne 'split' (train / val / test / excluded)."""
    df = df.copy()
    df["split"] = "excluded"                      # fichiers invalides : jamais utilisés
    usable = df[df["valid"]]

    test_idx = _pick_one_fold(usable, round(1 / test_ratio), seed)
    rest = usable.drop(test_idx)
    val_idx = _pick_one_fold(rest, round(len(rest) / (val_ratio * len(usable))), seed)

    df.loc[usable.index, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df.loc[val_idx, "split"] = "val"
    return df


# ======================= 3. CONTRÔLES =======================

def check_no_leakage(df: pd.DataFrame) -> None:
    """Lève une erreur si un même groupe apparaît dans plusieurs ensembles."""
    used = df[df["split"] != "excluded"]
    splits_per_group = used.groupby("group_id")["split"].nunique()
    leaking = splits_per_group[splits_per_group > 1]
    if not leaking.empty:
        raise ValueError(f"Data leakage : {len(leaking)} groupes dans plusieurs ensembles")
    logger.info("Aucune fuite : chaque groupe est dans un seul ensemble")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Taille de chaque ensemble et pourcentage de cas pathologiques."""
    used = df[df["split"] != "excluded"]
    summary = used.groupby("split").agg(n_audios=("label", "size"),
                                        n_groupes=("group_id", "nunique"),
                                        pct_abnormal=("label", "mean"))
    summary["pct_abnormal"] = (summary["pct_abnormal"] * 100).round(1)
    summary["pct_total"] = (summary["n_audios"] / len(used) * 100).round(1)
    return summary.reindex(["train", "val", "test"])