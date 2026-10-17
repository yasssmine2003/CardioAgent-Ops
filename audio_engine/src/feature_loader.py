"""Chargement des caractéristiques sauvegardées à l'étape 3 (data/features/)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_metadata() -> pd.DataFrame:
    """Une ligne par segment, dans le même ordre que les fichiers .npy."""
    return pd.read_csv(config.FEATURES_META_PATH)


def load_1d(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Vecteurs 1D (28 valeurs) d'un ensemble : (X, y, métadonnées)."""
    meta = load_metadata()
    mask = (meta["split"] == split).to_numpy()
    X = np.load(config.FEATURES_1D_PATH)[mask]
    return X, meta.loc[mask, "label"].to_numpy(), meta[mask].reset_index(drop=True)