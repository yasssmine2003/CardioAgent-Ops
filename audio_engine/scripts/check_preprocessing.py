"""Applique le prétraitement à tous les audios et construit l'index des segments.

Usage (depuis audio_engine/) :  python -m scripts.check_preprocessing

Produit :
- data/processed/segments_index.csv  : une ligne par segment de 5 s
- data/processed/excluded_audios.csv : les audios rejetés, avec la raison
Les signaux eux-mêmes ne sont pas sauvegardés : ils seront recalculés à l'étape 3.
"""
import logging
import time

import pandas as pd

from src.config import (EXCLUDED_PATH, INVENTORY_PATH, RAW_DIR, SEGMENT_HOP_S,
                        SEGMENTS_INDEX_PATH)
from src.preprocessing import AudioError, preprocess_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("check_preprocessing")
LOG_EVERY = 500
EXCLUDED_COLUMNS = ["path", "label", "source", "split", "group_id", "reason"]


def build_segments_index(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prétraite chaque audio ; retourne (index des segments, audios exclus)."""
    segments, excluded = [], []
    for i, row in enumerate(inventory.itertuples(index=False), start=1):
        info = {"path": row.path, "label": row.label, "source": row.source,
                "split": row.split, "group_id": row.group_id}
        try:
            result = preprocess_file(RAW_DIR / row.path)
        except AudioError as exc:
            excluded.append({**info, "reason": str(exc)})
        else:
            for k in range(len(result.segments)):
                segments.append({**info, "segment_idx": k,
                                 "start_s": k * SEGMENT_HOP_S, "padded": result.padded})
        if i % LOG_EVERY == 0:
            logger.info("%d / %d audios traités", i, len(inventory))
    return pd.DataFrame(segments), pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS)


def main() -> None:
    inventory = pd.read_csv(INVENTORY_PATH)
    inventory = inventory[inventory["split"] != "excluded"]

    start = time.perf_counter()
    seg, excl = build_segments_index(inventory)
    logger.info("Terminé en %.0f s", time.perf_counter() - start)

    seg.to_csv(SEGMENTS_INDEX_PATH, index=False)
    excl.to_csv(EXCLUDED_PATH, index=False)

    n_audios = seg["path"].nunique()
    print(f"\n=== Bilan ===")
    print(f"Audios conservés : {n_audios} | exclus : {len(excl)} | segments : {len(seg)}")
    print(f"Audios complétés par répétition : {seg.loc[seg['padded'], 'path'].nunique()}")

    print("\n=== Segments par ensemble et par classe ===")
    print(pd.crosstab(seg["split"], seg["label"], margins=True).reindex(["train", "val", "test", "All"]))
    print("\n% de segments pathologiques par ensemble :")
    print((seg.groupby("split")["label"].mean() * 100).round(1).reindex(["train", "val", "test"]))

    print("\n=== Audios exclus ===")
    if excl.empty:
        print("Aucun")
    else:
        print(pd.crosstab(excl["source"], excl["label"], margins=True))
        print("\nExemples de raisons :", excl["reason"].head(3).tolist())

    print(f"\nIndex écrit : {SEGMENTS_INDEX_PATH}")
    print(f"Exclusions  : {EXCLUDED_PATH}")


if __name__ == "__main__":
    main()