"""Décrit le contenu réel de data/raw/ (lecture seule, aucune modification)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from src.config import RAW_DIR

LABEL_EXT = {".csv", ".txt"}
ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz"}
MAX_EXAMPLES = 10
PHYSIONET_NAME = re.compile(r"^[a-f]\d{4}", re.IGNORECASE)  # ex : a0001.wav


def list_files() -> list[Path]:
    """Tous les fichiers de data/raw, sans les fichiers parasites de macOS."""
    return [p for p in RAW_DIR.rglob("*")
            if p.is_file() and "__MACOSX" not in p.parts and not p.name.startswith("._")]


def group_by_folder(files: list[Path]) -> dict[str, list[Path]]:
    """Regroupe les fichiers par dossier (chemin relatif à data/raw)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[str(p.parent.relative_to(RAW_DIR))].append(p)
    return groups


def show_folders(groups: dict[str, list[Path]]) -> None:
    """1. Nombre de fichiers par extension et exemples de noms."""
    print("===== 1. Dossiers et types de fichiers =====")
    for folder in sorted(groups):
        content = sorted(groups[folder])
        print(f"{folder:<40} {dict(Counter(p.suffix.lower() for p in content))}")
        print(f"    ex : {[p.name for p in content[:MAX_EXAMPLES]]}")


def guess_source(name: str) -> str:
    """Indice de provenance d'après le nom du fichier (à confirmer)."""
    if PHYSIONET_NAME.match(name):
        return "type PhysioNet (a0001...)"
    if "__" in name:
        return "type PASCAL (xxx__...)"
    return "autre"


def show_sources(groups: dict[str, list[Path]]) -> None:
    """2. Provenance probable des fichiers audio, par dossier."""
    print("\n===== 2. Provenance probable (d'après les noms) =====")
    for folder in sorted(groups):
        wavs = [p for p in groups[folder] if p.suffix.lower() == ".wav"]
        if wavs:
            print(f"{folder:<40} {dict(Counter(guess_source(p.name) for p in wavs))}")


def show_audio_stats(groups: dict[str, list[Path]]) -> None:
    """3. Fréquences d'échantillonnage, canaux et durées, par dossier."""
    print("\n===== 3. Audio : fréquences, canaux, durées =====")
    for folder in sorted(groups):
        rates, channels, durations, errors = Counter(), Counter(), [], 0
        for wav in (p for p in groups[folder] if p.suffix.lower() == ".wav"):
            try:
                info = sf.info(wav)
                rates[info.samplerate] += 1
                channels[info.channels] += 1
                durations.append(info.frames / info.samplerate)
            except Exception:
                errors += 1
        if not durations and not errors:
            continue
        print(f"{folder}")
        print(f"    fichiers lisibles={len(durations)}  illisibles={errors}")
        print(f"    fréquences (Hz)={dict(rates)}  canaux={dict(channels)}")
        if durations:
            print(f"    durée : min={min(durations):.1f}s  max={max(durations):.1f}s  "
                  f"moyenne={sum(durations) / len(durations):.1f}s")


def show_other_files(files: list[Path]) -> None:
    """4. Fichiers de labels possibles et archives non extraites."""
    print("\n===== 4. Autres fichiers =====")
    others = [p for p in files if p.suffix.lower() in LABEL_EXT | ARCHIVE_EXT]
    if not others:
        print("Aucun fichier .csv/.txt ni archive")
    for p in sorted(others)[:20]:
        print(f"    {p.relative_to(RAW_DIR)}")


def main() -> None:
    files = list_files()
    if not files:
        print(f"Aucun fichier dans {RAW_DIR} : vérifie l'étape 2.")
        return
    print(f"Total : {len(files)} fichiers dans {RAW_DIR}\n")
    groups = group_by_folder(files)
    show_folders(groups)
    show_sources(groups)
    show_audio_stats(groups)
    show_other_files(files)


if __name__ == "__main__":
    main()