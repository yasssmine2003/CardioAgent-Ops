"""Configuration centrale du Service 3 (audio-engine).

Tous les chemins sont relatifs à l'emplacement de ce fichier :
aucun chemin absolu, pour que le code tourne sur n'importe quel PC,
sur Colab ou dans Docker.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------- Chemins ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = Path(os.getenv("AUDIO_ENGINE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"              # données brutes, jamais modifiées
PROCESSED_DIR: Path = DATA_DIR / "processed"  # inventaire, splits, index des segments
FEATURES_DIR: Path = DATA_DIR / "features"    # caractéristiques calculées
MODELS_DIR: Path = PROJECT_ROOT / "models"

DATASET_DIR: Path = RAW_DIR / "ourdataset" / "train"
INVENTORY_PATH: Path = PROCESSED_DIR / "inventory.csv"
SEGMENTS_INDEX_PATH: Path = PROCESSED_DIR / "segments_index.csv"
EXCLUDED_PATH: Path = PROCESSED_DIR / "excluded_audios.csv"

# ---------- Labels ----------
# Nom du dossier -> label numérique (1 = classe pathologique)
LABELS: dict[str, int] = {"normal": 0, "abnormal": 1}

# ---------- Sources ----------
# Identifiées par la fréquence d'échantillonnage (vérifié lors de l'inspection).
SOURCE_BY_SAMPLE_RATE: dict[int, str] = {
    2000: "physionet2016",
    4000: "pascal_B",
    44100: "pascal_A",
}

# ---------- Découpage train / val / test ----------
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
RANDOM_SEED: int = 42

# ---------- Prétraitement ----------
TARGET_SR: int = 2000                 # [A] rééchantillonnage imposé par le cahier des charges
SEGMENT_DURATION_S: float = 5.0       # [A] 2 s ou 5 s -> choix 5 s (EDA, section 5)
SEGMENT_HOP_S: float = 5.0            # [C] pas entre deux segments (= durée : pas de chevauchement)
MIN_DURATION_S: float = 2.0           # [C] en dessous : audio exclu (moins de 2 battements)
SILENCE_STD_THRESHOLD: float = 1e-4   # [C] écart-type minimal ; en dessous = audio silencieux

# ---------- Caractéristiques (features) ----------
N_FFT: int = 256                      # [C] fenêtre d'analyse : 256 échantillons = 128 ms à 2000 Hz
HOP_LENGTH: int = 64                  # [C] pas entre deux fenêtres : 64 échantillons = 32 ms
N_MELS: int = 32                      # [C] nombre de bandes Mel (spectrogramme Mel = image 32 x 157)
N_MFCC: int = 13                      # [C] nombre de coefficients MFCC (valeur standard)
FMAX: float = TARGET_SR / 2           # fréquence max analysable (Nyquist) = 1000 Hz

FEATURES_1D_PATH: Path = FEATURES_DIR / "features_1d.npy"       # (N, 28)      -> DT, SVM
MEL_PATH: Path = FEATURES_DIR / "mel_spectrograms.npy"         # (N, 32, 157) -> CNN, ResNet, Transformer
FEATURES_META_PATH: Path = FEATURES_DIR / "metadata.csv"        # une ligne par segment (même ordre)
FEATURES_INFO_PATH: Path = FEATURES_DIR / "features_info.json"  # paramètres utilisés



# ---------- Évaluation et sélection ----------
DECISION_THRESHOLD: float = 0.5      # probabilité au-dessus de laquelle on prédit "pathologique"
MIN_SPECIFICITY: float = 0.70        # [C] garde-fou : au moins 70 % des normaux bien classés

# ---------- MLflow ----------
MLFLOW_EXPERIMENT: str = "cardioagent-audio-engine"
# Par défaut : dossier mlruns/ local. Sur Colab : redéfini vers Google Drive.
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri())


# ---------- Deep learning ----------
BATCH_SIZE: int = 64                 # [C] segments par lot d'entraînement
MAX_EPOCHS: int = 40                 # [C] nombre maximal de passages sur le train
PATIENCE: int = 8                   # [C] arrêt anticipé après 8 époques sans amélioration
WEIGHT_DECAY: float = 1e-4           # [C] régularisation L2 (limite le sur-apprentissage)


# ---------- Benchmark et sélection ----------
REPORTS_DIR: Path = DATA_DIR / "reports"                 # résultats du benchmark (sur Drive avec Colab)
BENCHMARK_CSV_PATH: Path = REPORTS_DIR / "benchmark.csv"
BEST_MODEL_PATH: Path = REPORTS_DIR / "best_model.json"  # modèle sélectionné + seuil -> export ONNX
ALT_SPECIFICITY: float = 0.80                            # [C] analyse de sensibilité : exigence plus forte
BENCHMARK_MODELS: tuple[str, ...] = ("decision_tree", "svm", "cnn", "resnet18", "transformer_cnn")