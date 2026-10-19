"""Registre des modèles : nom -> module (chaque module expose PARAM_GRID et build_model).

Les modèles de deep learning ne sont importés que si PyTorch est installé :
les modèles classiques restent utilisables sans lui.
"""
from src.models import decision_tree, svm

CLASSICAL_MODELS = {
    "decision_tree": decision_tree,
    "svm": svm,
}

try:
    from src.models import cnn, resnet, transformer_cnn
    DEEP_MODELS = {"cnn": cnn, "resnet18": resnet, "transformer_cnn": transformer_cnn}
except ImportError:  # PyTorch absent
    DEEP_MODELS = {}