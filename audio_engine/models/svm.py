"""Modèle 2 — SVM à noyau RBF sur caractéristiques 1D (MFCC + ZCR)."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_SEED

PARAM_GRID = [
    {"C": c, "gamma": gamma}
    for c in (1.0, 10.0)
    for gamma in ("scale", 0.01)
]


def build_model(C: float, gamma: float | str) -> Pipeline:
    """Standardisation + SVM RBF, dans un seul objet (Pipeline).

    - StandardScaler : moyenne 0 / écart-type 1 pour chaque caractéristique, appris sur le
      train uniquement (le Pipeline le garantit) ; indispensable car le SVM compare des distances
    - C              : tolérance aux erreurs (petit = frontière souple, grand = frontière stricte)
    - gamma          : portée de l'influence d'un exemple (petit = frontière lisse)
    - probability    : True pour obtenir une probabilité (sortie du service, export ONNX)
    - class_weight   : "balanced" (rééquilibrage des classes)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                    class_weight="balanced", random_state=RANDOM_SEED)),
    ])