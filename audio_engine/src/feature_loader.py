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



def load_mel(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Mel-spectrogrammes d'un ensemble : (X de forme (N, 1, 32, 157) en float32, y, métadonnées).

    La dimension 1 est le « canal » : les réseaux de convolution attendent des images
    (canaux, hauteur, largeur) ; un spectrogramme est une image à un seul canal.
    """
    meta = load_metadata()
    mask = (meta["split"] == split).to_numpy()
    X = np.load(config.MEL_PATH, mmap_mode="r")[mask].astype(np.float32)[:, np.newaxis]
    return X, meta.loc[mask, "label"].to_numpy(), meta[mask].reset_index(drop=True)