"""Modèle 1 — Decision Tree sur caractéristiques 1D (MFCC + ZCR)."""
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED

# Combinaisons d'hyperparamètres testées (chacune = un run MLflow)
PARAM_GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf}
    for depth in (5, 10, 20)
    for leaf in (1, 20)
]


def build_model(max_depth: int, min_samples_leaf: int) -> DecisionTreeClassifier:
    """Arbre de décision.

    - max_depth        : nombre maximal de questions successives (limite le sur-apprentissage)
    - min_samples_leaf : nombre minimal d'exemples par feuille (évite les règles trop spécifiques)
    - class_weight     : "balanced" -> une erreur sur un malade compte ~3 fois plus (rééquilibrage)
    """
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  class_weight="balanced", random_state=RANDOM_SEED)