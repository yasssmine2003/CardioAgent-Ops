"""Prédit un ou plusieurs fichiers .wav avec le modèle ONNX déployé.

Usage (depuis audio_engine/) :
    python -m scripts.predict chemin/vers/fichier.wav [autre.wav ...]
"""
import argparse
import sys

from src.inference import CardioPredictor
from src.preprocessing import AudioError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", help="fichier(s) .wav à analyser")
    args = parser.parse_args()

    predictor = CardioPredictor()   # chargé une seule fois pour tous les fichiers
    status = 0
    for path in args.audio:
        try:
            print(predictor.predict(path).model_dump_json(indent=2))
        except AudioError as exc:
            print(f"{path} : audio inutilisable — {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())