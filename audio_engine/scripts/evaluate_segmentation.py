"""Évalue la segmentation (approche 2) avec les annotations de référence PhysioNet.

Usage (depuis audio_engine/) :
    python -m scripts.download_annotations --split val     # une seule fois
    python -m scripts.evaluate_segmentation --split val

Produit data/reports/cycles/segmentation_eval.csv et affiche les résultats :
- globaux, par classe et selon notre indicateur « fiable » ;
- par tranche de périodicité, pour régler le seuil SEG_MIN_PERIODICITY.
"""
import argparse
import logging

import pandas as pd

from src import config
from src.annotations import load_reference
from src.preprocessing import AudioError, load_audio, normalize_audio, resample_audio, validate_signal
from src.segmentation import segment_cycles
from src.segmentation_eval import compare

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger("evaluate_segmentation")
LABEL_NAMES = {0: "normal", 1: "pathologique"}
METRICS = ["hr_error_pct", "beat_precision", "landmark_on_s1", "cycles_ok", "cycles_missed", "cycles_split"]


def summary(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Médiane des mesures + part des enregistrements dont le rythme est juste à 10 % près."""
    grouped = (df.groupby(by, observed=True) if by is not None
               else df.assign(tous="tous").groupby("tous"))
    table = grouped[METRICS].median()
    table.insert(0, "enregistrements", grouped.size())
    table.insert(1, "rythme_juste_10pct", grouped["hr_error_pct"].apply(lambda s: round((s <= 10).mean() * 100, 1)))
    return table.round(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["val"], choices=["train", "val"])
    args = parser.parse_args()

    inventory = pd.read_csv(config.INVENTORY_PATH)
    records = inventory[(inventory["source"] == "physionet2016") & inventory["split"].isin(args.split)]
    rows, no_annotation = [], 0
    for row in records.itertuples(index=False):
        name = row.path.split("/")[-1].removesuffix(".wav")
        mat = config.ANNOTATIONS_DIR / f"{name}_StateAns.mat"
        ref = load_reference(mat) if mat.is_file() else None
        if ref is None or len(ref.s1_starts) < 3:
            no_annotation += 1                        # absent, vide (« unsure ») ou trop court
            continue
        try:
            y, sr = load_audio(config.RAW_DIR / row.path)
            seg = segment_cycles(normalize_audio(resample_audio(validate_signal(y), sr)))
        except AudioError as exc:
            logger.warning("%s ignoré : %s", name, exc)
            continue
        rows.append({"name": name, "classe": LABEL_NAMES[row.label], "periodicity": seg.periodicity,
                     "reliable": seg.reliable, **compare(seg, ref)})

    df = pd.DataFrame(rows)
    config.CYCLES_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CYCLES_REPORT_DIR / "segmentation_eval.csv", index=False)
    df["tranche_periodicite"] = pd.cut(df["periodicity"], [0, 0.2, 0.3, 0.4, 1.0],
                                       labels=["< 0,2", "0,2-0,3", "0,3-0,4", ">= 0,4"])

    pd.set_option("display.width", 220)
    print(f"\nEnregistrements évalués : {len(df)} (sans annotation exploitable : {no_annotation})")
    print("\n=== Global ===");                       print(summary(df).to_string())
    print("\n=== Par classe ===");                   print(summary(df, "classe").to_string())
    print("\n=== Selon notre indicateur « fiable » ==="); print(summary(df, "reliable").to_string())
    print("\n=== Par tranche de périodicité ===");     print(summary(df, "tranche_periodicite").to_string())


if __name__ == "__main__":
    main()