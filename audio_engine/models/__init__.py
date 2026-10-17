"""Registre des modèles : nom -> module (chaque module expose PARAM_GRID et build_model)."""
from src.models import decision_tree, svm

CLASSICAL_MODELS = {
    "decision_tree": decision_tree,
    "svm": svm,
}