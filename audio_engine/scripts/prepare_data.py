"""Construit data/processed/inventory.csv : inventaire + découpage train/val/test.

Usage (depuis audio_engine/) :  python -m scripts.prepare_data
"""
import logging

import pandas as pd

from src.config import INVENTORY_PATH
from src.datasets import assign_splits, build_inventory, check_no_leakage, summarize

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    df = assign_splits(build_inventory())
    check_no_leakage(df)

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY_PATH, index=False)

    used = df[df["split"] != "excluded"]
    print("\n=== Répartition ===")
    print(summarize(df))
    print("\n=== Audios par source et par ensemble ===")
    print(pd.crosstab(used["source"], used["split"], margins=True))
    print("\n=== Taille des groupes (nb de groupes contenant N audios) ===")
    print(df.groupby("group_id").size().value_counts().sort_index())
    print(f"\nInventaire écrit : {INVENTORY_PATH}")


if __name__ == "__main__":
    main()