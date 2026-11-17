"""Télécharge les annotations PhysioNet (S1, systole, S2, diastole) corrigées à la main.

Usage (depuis audio_engine/) :  python -m scripts.download_annotations --split val

Seuls les enregistrements PhysioNet de l'ensemble demandé sont téléchargés (fichiers
d'environ 1 Ko). Le test n'est jamais utilisé pour régler la segmentation.
"""
import argparse
import logging
import time
import urllib.request

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger("download_annotations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["val"], choices=["train", "val"])
    args = parser.parse_args()

    inventory = pd.read_csv(config.INVENTORY_PATH)
    records = inventory[(inventory["source"] == "physionet2016") & inventory["split"].isin(args.split)]
    config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0
    for path in records["path"]:
        name = path.split("/")[-1].removesuffix(".wav")
        target = config.ANNOTATIONS_DIR / f"{name}_StateAns.mat"
        if target.is_file():
            skipped += 1
            continue
        url = config.ANNOTATIONS_URL.format(subset=name[0], name=name)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
                downloaded += 1
                break
            except Exception as exc:
                if attempt == 2:
                    logger.warning("%s : échec (%s)", name, exc)
                    failed += 1
                time.sleep(2)
    logger.info("Téléchargés : %d | déjà présents : %d | échecs : %d", downloaded, skipped, failed)


if __name__ == "__main__":
    main()