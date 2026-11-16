"""Construit les caractéristiques « cycles » (fenêtres alignées sur les battements).

Usage (depuis audio_engine/, sur le PC) :
    set AUDIO_ENGINE_FEATURE_SET=cycles        (Windows)
    python -m scripts.build_cycle_windows

Produit dans data/features_cycles/ (même format que data/features/) :
- features_1d.npy, mel_spectrograms.npy, metadata.csv, features_info.json
- rhythm_features.csv : 1 ligne par enregistrement (fréquence cardiaque, irrégularité...) -> Service 4
Même split que avant (inventory.csv) : la comparaison fixe / cycles est donc équitable.
"""
import json
import logging
import time

import numpy as np
import pandas as pd

from src import config
from src.cycle_windows import build_windows
from src.features import FEATURE_NAMES, MEL_SHAPE, N_FRAMES, extract_features
from src.preprocessing import AudioError, load_audio, normalize_audio, resample_audio, validate_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger("build_cycle_windows")
LOG_EVERY = 250


def main() -> None:
    if config.FEATURE_SET != "cycles":
        raise SystemExit("Mettre d'abord AUDIO_ENGINE_FEATURE_SET=cycles (sinon on écraserait data/features/)")

    inventory = pd.read_csv(config.INVENTORY_PATH)
    inventory = inventory[inventory["split"] != "excluded"]

    rows, rhythm, feats_1d, mels = [], [], [], []
    start = time.perf_counter()
    for i, r in enumerate(inventory.itertuples(index=False), start=1):
        info = {"path": r.path, "label": r.label, "source": r.source, "split": r.split, "group_id": r.group_id}
        try:
            y, sr = load_audio(config.RAW_DIR / r.path)
            y = normalize_audio(resample_audio(validate_signal(y), sr))
            segments, windows, seg = build_windows(y)
        except AudioError:
            continue                                       # < 2 s : exclu, comme avant
        for k, (segment, w) in enumerate(zip(segments, windows)):
            v, m = extract_features(segment)
            feats_1d.append(v)
            mels.append(m.astype(np.float16))
            rows.append({**info, "segment_idx": k, "start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3),
                         "n_cycles": w.n_cycles, "mode": w.mode,
                         "padded": (w.end_s - w.start_s) < config.SEGMENT_DURATION_S - 1e-3})
        rhythm.append({**info, "heart_rate_bpm": seg.heart_rate_bpm, "periodicity": seg.periodicity,
                       "irregularity": seg.irregularity, "n_valid_cycles": len(seg.valid_cycles),
                       "reliable": seg.reliable, "mode": windows[0].mode})
        if i % LOG_EVERY == 0:
            logger.info("%d / %d audios traités", i, len(inventory))
    logger.info("Terminé en %.0f s", time.perf_counter() - start)

    feats_1d, mels = np.stack(feats_1d).astype(np.float32), np.stack(mels)
    if not (np.isfinite(feats_1d).all() and np.isfinite(mels).all()):
        raise RuntimeError("Valeurs NaN/Inf détectées")
    assert mels.shape[1:] == MEL_SHAPE

    meta, rhythm = pd.DataFrame(rows), pd.DataFrame(rhythm)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.FEATURES_1D_PATH, feats_1d)
    np.save(config.MEL_PATH, mels)
    meta.to_csv(config.FEATURES_META_PATH, index=False)
    rhythm.to_csv(config.FEATURES_DIR / "rhythm_features.csv", index=False)
    config.FEATURES_INFO_PATH.write_text(json.dumps({
        "feature_set": "cycles", "sample_rate": config.TARGET_SR,
        "segment_duration_s": config.SEGMENT_DURATION_S, "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH, "n_mels": config.N_MELS, "n_mfcc": config.N_MFCC,
        "fmax": config.FMAX, "n_frames": N_FRAMES, "features_1d_shape": list(feats_1d.shape),
        "mel_shape": list(mels.shape), "feature_names": FEATURE_NAMES}, indent=2), encoding="utf-8")

    print("\n=== Bilan ===")
    print(f"Audios : {meta['path'].nunique()} | fenêtres : {len(meta)}")
    print(f"Audios en mode cycles : {(rhythm['mode'] == 'cycles').mean():.0%} (le reste : repli fenêtres fixes)")
    print(f"Cycles par fenêtre (médiane) : {meta.loc[meta['mode'] == 'cycles', 'n_cycles'].median():.0f}")
    print("\nFenêtres par ensemble et par classe :")
    print(pd.crosstab(meta["split"], meta["label"], margins=True).reindex(["train", "val", "test", "All"]))
    print(f"\nFichiers écrits dans : {config.FEATURES_DIR}")


if __name__ == "__main__":
    main()
