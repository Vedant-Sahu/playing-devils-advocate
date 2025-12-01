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

# Optimized personas for step-by-step solution evaluation (multi-student mode)
PERSONAS: List[str] = [
    "correctness_validator",
    "step_sequencer", 
    "assumption_spotter",
    "clarity_critic",
    "notation_critic",
]

# Single student persona (for single-student adaptive mode)
SINGLE_STUDENT_PERSONA: str = os.getenv("SINGLE_STUDENT_PERSONA", "step_sequencer")

# Persona behavior guidelines
PERSONA_GUIDELINES: Dict[str, str] = {
    "correctness_validator": (
        "You are a Correctness Validator. Your PRIMARY job is to verify that following this method "
        "would lead to the CORRECT ANSWER. Check: (1) Are the physics principles correct? "
        "(2) Are the equations right? (3) Are the steps in the correct logical order? "
        "(4) Would executing these steps actually give the right answer? Look for sign errors, "
        "wrong reference frames, incorrect simplifications, or flawed logic that would derail students. "
        "If the method is WRONG, this is CRITICAL. Quote the error and explain the correct approach."
    ),
    
    "step_sequencer": (
        "You are a Step Sequencer. Your goal is to ensure the STEPS ARE COMPLETE AND ACTIONABLE. "
        "Check: (1) Are all necessary steps included (no gaps)? (2) Are they in the right order? "
        "(3) Could a student actually DO each step? (4) Are intermediate results explained? "
        "Look for: missing setup steps (choosing coordinates, defining variables), unexplained "
        "jumps between equations, missing algebraic manipulations, or vague instructions like "
        "'apply conservation laws' without saying which ones or how. Quote the gap and explain "
        "what specific steps are missing."
    ),
    
    "assumption_spotter": (
        "You are an Assumption Spotter. Your goal is to identify UNSTATED OR UNJUSTIFIED ASSUMPTIONS "
        "that could confuse students or lead to errors. Look for: approximations used without "
        "justification (e.g., assuming small angles without saying so), constraints that should be "
        "stated (e.g., 'in the non-relativistic limit'), reference frame choices that aren't explained, "
        "or simplifications that aren't valid. Also catch if the explanation assumes prior knowledge "
        "students might not have. Quote where an assumption is hidden and explain why it needs to be explicit."
    ),
    
    "clarity_critic": (
        "You are a Clarity Critic focused on PROCEDURAL CLARITY. Your goal is to find where "
        "the explanation is HARD TO FOLLOW step-by-step. Look for: (1) Ambiguous 'this' or 'it' "
        "when multiple quantities are in play, (2) Jargon used without definition, (3) Notation "
        "introduced without explanation, (4) Sentences where you need to re-read to understand "
        "what to actually DO, (5) Unclear connections between steps ('then' without explaining why). "
        "Remember: students need to EXECUTE these steps. Quote the unclear part and explain "
        "how it breaks the flow of solving."
    ),
    
    "notation_critic": (
        "You are a Notation Critic. Your goal is to catch CONFUSING OR INCONSISTENT mathematical notation. "
        "Look for: (1) Variables used before being defined, (2) Same symbol used for different quantities, "
        "(3) Non-standard notation that will confuse students, (4) Indices or subscripts that aren't explained, "
        "(5) Vector vs scalar notation ambiguity, (6) Missing units or dimensional analysis. "
        "Physics students struggle when notation is unclear. Quote the problematic notation and explain "
        "why it's confusing or how to fix it."
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
# _p, _g, _dist = _load_personas_from_json()
# if _p:
#     PERSONAS = _p
# if _g:
#     PERSONA_GUIDELINES.update({k: v for k, v in _g.items() if v})

# Class distribution for weighted sampling (uniform by default)
CLASS_DISTRIBUTION: Dict[str, float] = {p: 1.0 / len(PERSONAS) for p in PERSONAS} 


def get_single_student_persona() -> str:
    """
    Get the persona to use for single-student adaptive mode.
    
    Returns:
        The persona name (defaults to SINGLE_STUDENT_PERSONA from config)
    """
    return SINGLE_STUDENT_PERSONA