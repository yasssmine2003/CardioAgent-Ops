"""Extrait les caractéristiques de tous les segments et les sauvegarde dans data/features/.

Usage (depuis audio_engine/) :  python -m scripts.extract_features

Produit (la ligne i de chaque fichier correspond au même segment) :
- features_1d.npy       (N, 28)       float32 -> Decision Tree, SVM
- mel_spectrograms.npy  (N, 32, 157)  float16 -> CNN, ResNet18, Transformer-CNN
- metadata.csv          N lignes : audio d'origine, label, split, source...
- features_info.json    paramètres utilisés (traçabilité + inférence)
"""
import json
import logging
import time

import numpy as np
import pandas as pd

from src import config
from src.features import FEATURE_NAMES, MEL_SHAPE, N_FRAMES, extract_features
from src.preprocessing import preprocess_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("extract_features")
LOG_EVERY = 500


def extract_all(index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prétraite chaque audio une seule fois, puis extrait les caractéristiques de ses segments.

    La ligne i des tableaux retournés correspond à la ligne i de `index`.
    """
    features_1d = np.zeros((len(index), len(FEATURE_NAMES)), dtype=np.float32)
    mels = np.zeros((len(index), *MEL_SHAPE), dtype=np.float16)  # float16 : 2 fois moins de place

    groups = index.groupby("path", sort=False)
    for i, (path, rows) in enumerate(groups, start=1):
        segments = preprocess_file(config.RAW_DIR / path).segments
        if len(segments) != len(rows):  # contrôle de cohérence avec l'étape 2
            raise RuntimeError(f"{path} : {len(segments)} segments au lieu de {len(rows)}")
        for position, segment_idx in zip(rows.index, rows["segment_idx"]):
            features_1d[position], mels[position] = extract_features(segments[segment_idx])
        if i % LOG_EVERY == 0:
            logger.info("%d / %d audios traités", i, groups.ngroups)
    return features_1d, mels


def save_info(features_1d: np.ndarray, mels: np.ndarray) -> None:
    """Enregistre les paramètres d'extraction : indispensable pour reproduire à l'inférence."""
    info = {
        "sample_rate": config.TARGET_SR,
        "segment_duration_s": config.SEGMENT_DURATION_S,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "n_mels": config.N_MELS,
        "n_mfcc": config.N_MFCC,
        "fmax": config.FMAX,
        "n_frames": N_FRAMES,
        "features_1d_shape": list(features_1d.shape),
        "mel_shape": list(mels.shape),
        "feature_names": FEATURE_NAMES,
    }
    config.FEATURES_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    index = pd.read_csv(config.SEGMENTS_INDEX_PATH)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    features_1d, mels = extract_all(index)
    logger.info("Terminé en %.0f s", time.perf_counter() - start)

    if not (np.isfinite(features_1d).all() and np.isfinite(mels).all()):
        raise RuntimeError("Valeurs NaN/Inf détectées dans les caractéristiques")

    np.save(config.FEATURES_1D_PATH, features_1d)
    np.save(config.MEL_PATH, mels)
    index.to_csv(config.FEATURES_META_PATH, index=False)
    save_info(features_1d, mels)

    print("\n=== Bilan ===")
    print(f"Vecteurs 1D         : {features_1d.shape}  ({config.FEATURES_1D_PATH.stat().st_size / 1e6:.1f} Mo)")
    print(f"Mel-spectrogrammes  : {mels.shape}  ({config.MEL_PATH.stat().st_size / 1e6:.1f} Mo)")
    print(f"Segments par split  : {index['split'].value_counts().reindex(['train', 'val', 'test']).to_dict()}")
    print("\nAperçu des valeurs (vecteurs 1D, ensemble train) :")
    train_1d = pd.DataFrame(features_1d[index["split"] == "train"], columns=FEATURE_NAMES)
    print(train_1d[["mfcc_0_mean", "mfcc_1_mean", "zcr_mean"]].describe().round(3))
    print(f"\nMel (dB) : min={mels.min():.1f}  max={mels.max():.1f}  moyenne={mels.astype(np.float32).mean():.1f}")
    print(f"\nFichiers écrits dans : {config.FEATURES_DIR}")


if __name__ == "__main__":
    main()