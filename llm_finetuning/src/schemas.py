"""Schéma Pydantic de la réponse du LLM à un résultat du Service 3 (garde-fou anti-hallucination [A]).

Le Service 4 peut réutiliser ce schéma pour valider chaque réponse de llm-server :
    InterpretationLLM.model_validate_json(texte)   -> lève ValidationError si le format n'est pas respecté
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InterpretationLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")          # aucun champ inventé n'est accepté

    resultat: Literal["normal", "pathologique"]
    fiabilite: Literal["faible", "moyenne", "élevée"]
    cas_limite: bool
    interpretation: str = Field(min_length=10)
    recommandation: str = Field(min_length=10)
    avertissement: str = Field(min_length=10)


def parse_llm_answer(text: str) -> InterpretationLLM:
    """Extrait le JSON de la réponse (le modèle peut ajouter du texte autour) puis le valide."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Aucun objet JSON dans la réponse")
    return InterpretationLLM.model_validate(json.loads(text[start:end + 1]))
