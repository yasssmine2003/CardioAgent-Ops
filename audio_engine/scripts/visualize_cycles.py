"""Applique la segmentation par cycles à de vrais enregistrements : figures + statistiques.

Usage (depuis audio_engine/) :
    python -m scripts.visualize_cycles                    # 5 figures par classe, statistiques sur 100 par classe
    python -m scripts.visualize_cycles --plots 3 --stats 300
    python -m scripts.visualize_cycles --files ourdataset/train/normal/e00679.wav   # figures de fichiers précis

Produit dans data/reports/cycles/ :
- une figure par enregistrement illustré (signal, cycles détectés, enveloppe)
- cycles_stats.csv : une ligne par enregistrement analysé
On n'utilise PAS l'ensemble de test (il reste réservé à l'évaluation finale).
"""
import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import AudioError, load_audio, normalize_audio, resample_audio, validate_signal
from src.segmentation import CycleSegmentation, homomorphic_envelope, segment_cycles

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger("visualize_cycles")
PLOT_SECONDS = 8.0
LABEL_NAMES = {0: "normal", 1: "pathologique"}


def prepare(path) -> np.ndarray:
    """Même prétraitement que le service : mono, 2000 Hz, z-score."""
    y, sr = load_audio(path)
    return normalize_audio(resample_audio(validate_signal(y), sr))


def plot_recording(y: np.ndarray, seg: CycleSegmentation, title: str, out_path) -> None:
    """Haut : signal prétraité. Bas : enveloppe, battements et cycles (vert = valide, rouge = rejeté, gris = coupure)."""
    sr, env_sr = config.TARGET_SR, config.SEG_ENVELOPE_SR
    n = int(min(len(y), PLOT_SECONDS * sr))
    env = homomorphic_envelope(y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    ax1.plot(np.arange(n) / sr, y[:n], lw=0.3, color="0.35")
    ax1.set(ylabel="Signal (z-score)", title=title)
    for start, end in seg.silences:                        # coupures : zones grises
        for ax in (ax1, ax2):
            ax.axvspan(start, min(end, PLOT_SECONDS), color="0.6", alpha=0.35)
    for c in seg.cycles:                                   # cycles : vert = valide, rouge = rejeté
        if c.start_s < PLOT_SECONDS:
            color = "tab:green" if c.valid else "tab:red"
            ax2.axvspan(c.start_s, min(c.end_s, PLOT_SECONDS), color=color, alpha=0.12)
            ax2.axvline(c.start_s, color=color, lw=0.8)
    m = int(min(len(env), PLOT_SECONDS * env_sr))
    ax2.plot(np.arange(m) / env_sr, env[:m], color="tab:blue", lw=1)
    beats = seg.beat_times_s[seg.beat_times_s < PLOT_SECONDS]
    ax2.plot(beats, np.interp(beats, np.arange(len(env)) / env_sr, env), "v", color="tab:orange",
             label="battements détectés")
    ax2.set(xlabel="Temps (s)", ylabel="Enveloppe", xlim=(0, n / sr))
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summarize(row, seg: CycleSegmentation) -> dict:
    n_cycles = len(seg.cycles)
    return {"path": row.path, "label": row.label, "source": row.source, "duration_s": seg.duration_s,
            "heart_rate_bpm": seg.heart_rate_bpm, "periodicity": seg.periodicity,
            "n_cycles": n_cycles, "n_valid": len(seg.valid_cycles),
            "pct_valid": round(len(seg.valid_cycles) / n_cycles * 100, 1) if n_cycles else 0.0,
            "irregularity": seg.irregularity, "reliable": seg.reliable,
            "silence_s": seg.silence_s,
            "rejected_duration": sum(c.reason == "duree" for c in seg.cycles),
            "rejected_silence": sum(c.reason == "coupure" for c in seg.cycles)}


def title_for(row, seg: CycleSegmentation) -> str:
    name = row.path.split("/")[-1].removesuffix(".wav")
    silence = f", coupures {seg.silence_s} s" if seg.silences else ""
    return (f"{name} — {LABEL_NAMES[row.label]} ({row.source}) — {seg.heart_rate_bpm} bpm, "
            f"périodicité {seg.periodicity}, {len(seg.valid_cycles)}/{len(seg.cycles)} cycles valides{silence}")


def plot_files(inventory: pd.DataFrame, files: list[str]) -> None:
    """Trace la figure de fichiers précis (chemins relatifs à data/raw, comme dans inventory.csv)."""
    config.CYCLES_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        rows = inventory[inventory["path"] == path.replace("\\", "/")]
        if rows.empty:
            logger.warning("%s absent de l'inventaire", path)
            continue
        row = next(rows.itertuples(index=False))
        y = prepare(config.RAW_DIR / row.path)
        seg = segment_cycles(y)
        name = row.path.split("/")[-1].removesuffix(".wav")
        out = config.CYCLES_REPORT_DIR / f"{LABEL_NAMES[row.label]}_{name}.png"
        plot_recording(y, seg, title_for(row, seg), out)
        print(f"{path} -> {seg.heart_rate_bpm} bpm, périodicité {seg.periodicity}, "
              f"{len(seg.valid_cycles)}/{len(seg.cycles)} cycles valides, coupures {seg.silence_s} s -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", type=int, default=5, help="figures par classe")
    parser.add_argument("--stats", type=int, default=100, help="enregistrements analysés par classe")
    parser.add_argument("--files", nargs="+", help="tracer seulement ces fichiers (chemins de inventory.csv)")
    args = parser.parse_args()

    if args.files:
        plot_files(pd.read_csv(config.INVENTORY_PATH), args.files)
        return

    inventory = pd.read_csv(config.INVENTORY_PATH)
    pool = inventory[inventory["split"].isin(["train", "val"])]          # le test reste intact
    sample = pd.concat([pool[pool["label"] == label].sample(
        min(args.stats, int((pool["label"] == label).sum())), random_state=config.RANDOM_SEED)
        for label in LABEL_NAMES])
    config.CYCLES_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows, plotted = [], {0: 0, 1: 0}
    for row in sample.itertuples(index=False):
        try:
            y = prepare(config.RAW_DIR / row.path)
        except AudioError as exc:
            logger.warning("%s ignoré : %s", row.path, exc)
            continue
        seg = segment_cycles(y)
        rows.append(summarize(row, seg))
        if plotted[row.label] < args.plots:
            name = row.path.split("/")[-1].removesuffix(".wav")
            plot_recording(y, seg, title_for(row, seg), config.CYCLES_REPORT_DIR / f"{LABEL_NAMES[row.label]}_{name}.png")
            plotted[row.label] += 1

    stats = pd.DataFrame(rows)
    stats.to_csv(config.CYCLES_REPORT_DIR / "cycles_stats.csv", index=False)

    def table(by):
        return stats.groupby(by).agg(
            enregistrements=("path", "size"),
            fiables_pct=("reliable", lambda s: round(s.mean() * 100, 1)),
            rythme_median_bpm=("heart_rate_bpm", "median"),
            periodicite_mediane=("periodicity", "median"),
            cycles_valides_pct=("pct_valid", "median"),
            irregularite_mediane=("irregularity", "median"),
            avec_coupure_pct=("silence_s", lambda s: round((s > 0).mean() * 100, 1)),
        ).round(3)

    pd.set_option("display.width", 200)
    print("\n=== Par classe ===")
    print(table(stats["label"].map(LABEL_NAMES)).to_string())
    print("\n=== Par source ===")
    print(table("source").to_string())
    extremes = stats[(stats["heart_rate_bpm"] >= 130) | (stats["heart_rate_bpm"] <= 45)]
    print(f"\nRythmes extrêmes (>= 130 ou <= 45 bpm) : {len(extremes)}")
    if len(extremes):
        print(extremes[["path", "label", "heart_rate_bpm", "periodicity", "pct_valid"]].to_string(index=False))
    print(f"\nFigures et statistiques : {config.CYCLES_REPORT_DIR}")


if __name__ == "__main__":
    main()