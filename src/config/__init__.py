"""Configuration package for agent settings and personas."""

from .agent_config import (
    PERSONAS,
    PERSONA_GUIDELINES,
    CLASS_DISTRIBUTION,
    StopConfig,
    _llm,
    _model_for_role,
)

__all__ = [
    "PERSONAS",
    "PERSONA_GUIDELINES",
    "CLASS_DISTRIBUTION",
    "StopConfig",
    "_llm",
    "_model_for_role",
]