"""
Agent configuration including personas, LLM settings, and model selection.

This module provides:
- Student persona definitions and guidelines
- LLM model configuration and factory functions
- Dynamic persona loading from student_profiles.json
- Stopping configuration for iterative refinement
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Default personas used across the system
PERSONAS: List[str] = [
    "misconception_hunter",
    "clarity_critic",
    "rigor_checker",
    "completeness_auditor",
    "example_breaker",
]

# Persona behavior guidelines
PERSONA_GUIDELINES: Dict[str, str] = {
    "misconception_hunter": (
        "You are a Misconception Hunter. Your goal is to identify places where the explanation "
        "could CREATE or REINFORCE common physics misconceptions (e.g., impetus theory, "
        "force-velocity confusion, conflating correlation with causation). You are specifically "
        "looking for phrasing that could mislead students, even if technically correct. "
        "Be specific: quote the problematic phrase and explain which misconception it could create."
    ),
    
    "clarity_critic": (
        "You are a Clarity Critic. Your goal is to find the LEAST CLEAR part of the explanation. "
        "Look for: ambiguous pronouns, undefined jargon, logical gaps between sentences, "
        "missing intermediate steps, or sentences that require re-reading. You represent students "
        "who will give up if confused. Quote the unclear part and explain why it breaks comprehension flow."
    ),
    
    "rigor_checker": (
        "You are a Rigor Checker. Your goal is to find mathematical or logical ERRORS and "
        "unjustified claims. Look for: incorrect equations, sign errors, dimensional inconsistencies, "
        "unjustified assumptions, missing constraints, or logical leaps. You care about technical "
        "correctness. Quote the error and explain what's wrong technically."
    ),
    
    "completeness_auditor": (
        "You are a Completeness Auditor. Your goal is to identify the MOST IMPORTANT missing piece. "
        "What critical concept, step, definition, or context is omitted that would prevent full "
        "understanding? You're not looking for minor additions—find the ONE gap that matters most. "
        "Explain what's missing and why it's essential."
    ),
    
    "example_breaker": (
        "You are an Example Breaker. Your goal is to find EDGE CASES or scenarios where the "
        "explanation breaks down or becomes misleading. Think of boundary conditions, special cases, "
        "or counter-examples that would make students question the explanation. Quote the general "
        "claim and describe the case where it fails or misleads."
    ),
}

@dataclass
class StopConfig:
    """Configuration for stopping criteria in iterative refinement."""
    threshold: float = 0.7
    max_iterations: int = 5
    stagnation_window: int = 2
    stagnation_min_improvement: float = 0.02


def _model_for_role(role: str | None) -> str:
    """
    Get the appropriate model name for a given agent role.
    
    Checks environment variables for role-specific models, falling back
    to the default MODEL_NAME if not specified.
    
    Args:
        role: Agent role (e.g., "teacher", "student", "critique_eval")
        
    Returns:
        Model name string (e.g., "gpt-4o-mini")
    """
    default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    if role:
        key = {
            "teacher": "TEACHER_MODEL",
            "coordinator": "COORDINATOR_MODEL",
            "student": "STUDENT_MODEL",
            "critique_eval": "CRITIQUE_EVAL_MODEL",
            "answerer": "ANSWER_MODEL",
        }.get(role.lower().strip())
        if key:
            v = os.getenv(key)
            if v:
                return v
    return default_model


def _llm(
        temperature: float = 1.0, 
        json_mode: bool = False, 
        role: str | None = None, 
        max_tokens: int | None = None
    ) -> ChatOpenAI:
    """
    Create a configured ChatOpenAI instance.
    
    Args:
        temperature: Sampling temperature (0.0-1.0)
        json_mode: Whether to enforce JSON output format
        role: Agent role for model selection
        max_tokens: Maximum tokens for the model's completion
        
    Returns:
        Configured ChatOpenAI instance
    """
    model = _model_for_role(role)

    # Build common arguments
    kwargs = {
        "model": model,
        "temperature": temperature,
    }

    # Add max token limit if provided, unless disabled via env switch
    _disable_max = str(os.getenv("DISABLE_MAX_TOKENS", "")).strip().lower() in ("1", "true", "yes", "on")
    if (not _disable_max) and max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    # Add JSON mode formatting if requested
    name = str(model).lower()
    supports_json = (
        name.startswith("gpt-4o")
        or name.startswith("gpt-4.1")
        or name.startswith("o3")
        or name.startswith("o4")
    )
    if json_mode and supports_json:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatOpenAI(**kwargs)


def _load_personas_from_json() -> tuple[list[str] | None, dict[str, str] | None, dict[str, float] | None]:
    """
    Load personas and class distribution from data/student_profiles.json.
    
    Returns:
        Tuple of (personas list, guidelines dict, class distribution dict)
        Returns (None, None, None) if file not found or invalid
    """
    try:
        data_path = Path(__file__).resolve().parents[2] / "data" / "student_profiles.json"
        if not data_path.exists():
            return None, None, None
        data = json.loads(data_path.read_text(encoding="utf-8"))
        personas_json = data.get("personas", [])
        if not isinstance(personas_json, list) or not personas_json:
            return None, None, None
        personas: list[str] = []
        guides: dict[str, str] = {}
        for item in personas_json:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            guide = str(item.get("guidelines", "")).strip()
            if not name:
                continue
            personas.append(name)
            if guide:
                guides[name] = guide
        class_dist = data.get("class_distribution")
        if not personas:
            return None, None, None
        return personas, guides or None, class_dist if isinstance(class_dist, dict) else None
    except Exception:
        return None, None, None


# Load personas from JSON if available, otherwise use defaults
_p, _g, _dist = _load_personas_from_json()
if _p:
    PERSONAS = _p
if _g:
    PERSONA_GUIDELINES.update({k: v for k, v in _g.items() if v})

# Class distribution for weighted sampling (uniform by default)
CLASS_DISTRIBUTION: Dict[str, float] = (
    {p: 1.0 / len(PERSONAS) for p in PERSONAS} 
    if not _dist 
    else {str(k): float(v) for k, v in _dist.items() if str(k) in PERSONAS}
)