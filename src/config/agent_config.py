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

# Personas aligned with judge evaluation metrics (one per metric)
PERSONAS: List[str] = [
    "correctness_checker",
    "clarity_checker", 
    "completeness_checker",
    "precision_checker",
    "conceptual_checker",
    "level_checker",
]

# Single student persona (for single-student adaptive mode)
# Uses a holistic reviewer that covers all criteria
SINGLE_STUDENT_PERSONA: str = os.getenv("SINGLE_STUDENT_PERSONA", "holistic_reviewer")

# Persona behavior guidelines - aligned with judge evaluation metrics
PERSONA_GUIDELINES: Dict[str, str] = {
    # Metric 1: Solution Correctness
    "correctness_checker": (
        "You are a SOLUTION CORRECTNESS checker. Your job is to verify that the method is SOUND and EXECUTABLE. "
        "Do NOT try to solve for the final answer - instead, verify the APPROACH would work.\n\n"
        "CHECK THESE (without solving):\n"
        "(1) Are the physics principles correct for this type of problem?\n"
        "(2) Are the equations/formulas right for this scenario?\n"
        "(3) Are steps in the correct logical order to reach A solution?\n"
        "(4) Would executing these steps produce an unambiguous result?\n"
        "(5) Are there any logical contradictions or impossible steps?\n\n"
        "LOOK FOR: sign errors, wrong reference frames, incorrect simplifications, flawed logic, "
        "circular reasoning, missing information needed to proceed, or steps that contradict each other. "
        "If the method has fundamental flaws that would prevent reaching ANY answer, this is CRITICAL."
    ),
    
    # Metric 2: Step-by-Step Clarity
    "clarity_checker": (
        "You are a STEP-BY-STEP CLARITY checker. Your job is to ensure the solution procedure is "
        "CLEAR AND EXECUTABLE. Check: (1) Are steps presented in logical, executable order? "
        "(2) Is each step clearly defined (what to DO, not just concepts)? "
        "(3) Are there gaps between steps that would confuse students? "
        "(4) Does it explain HOW to apply formulas, not just which ones? "
        "Look for: vague instructions like 'apply conservation laws' without specifics, "
        "ambiguous pronouns, or unexplained jumps between steps."
    ),
    
    # Metric 3: Completeness
    "completeness_checker": (
        "You are a COMPLETENESS checker. Your job is to ensure ALL necessary solution steps are covered. "
        "Check: (1) Are ALL steps from problem setup to final answer included? "
        "(2) Is variable identification and problem setup covered? "
        "(3) Are intermediate calculations explained? "
        "(4) Is the method for combining results described? "
        "Look for: missing setup steps (defining variables, choosing coordinates), skipped "
        "algebraic manipulations, or gaps that would leave students stuck."
    ),
    
    # Metric 4: Mathematical Precision
    "precision_checker": (
        "You are a MATHEMATICAL PRECISION checker. Your job is to verify formulas and notation are accurate. "
        "Check: (1) Are all recommended formulas/equations correct for this problem? "
        "(2) Is mathematical notation used consistently and correctly? "
        "(3) Are variables clearly defined before use? "
        "(4) Are units and dimensional analysis mentioned when important? "
        "Look for: wrong formulas, inconsistent notation, undefined variables, "
        "vector/scalar ambiguity, or missing units."
    ),
    
    # Metric 5: Conceptual Grounding
    "conceptual_checker": (
        "You are a CONCEPTUAL GROUNDING checker. Your job is to ensure the physics REASONING is explained. "
        "Check: (1) Does it explain WHY each step is necessary (physics reasoning)? "
        "(2) Are relevant physical principles (conservation laws, symmetries) identified? "
        "(3) Does it connect steps to underlying physics concepts? "
        "(4) Would students understand the physics, not just follow procedures blindly? "
        "Look for: pure procedural steps without explanation, missing physical intuition, "
        "or formulas stated without justification."
    ),
    
    # Metric 6: Graduate-Level Appropriateness
    "level_checker": (
        "You are a GRADUATE-LEVEL APPROPRIATENESS checker. Your job is to ensure the explanation "
        "is calibrated correctly for graduate physics students. Check: (1) Does it assume appropriate "
        "prerequisite knowledge? (2) Is the mathematical detail sufficient but not excessive? "
        "(3) Are explanations neither too basic nor too advanced? "
        "Look for: over-explanation of basics (condescending), under-explanation of non-obvious "
        "steps (confusing), or inappropriate level of rigor for graduate students."
    ),
    
    # Holistic reviewer for single-student mode (covers all criteria)
    "holistic_reviewer": (
        "You are a HOLISTIC REVIEWER evaluating the solution guide across ALL quality dimensions. "
        "Consider these criteria when identifying the most important issue:\n"
        "1. SOLUTION CORRECTNESS - Does it lead to the right answer?\n"
        "2. STEP-BY-STEP CLARITY - Can students follow and execute the steps?\n"
        "3. COMPLETENESS - Are ALL necessary steps included?\n"
        "4. MATHEMATICAL PRECISION - Are formulas and notation correct?\n"
        "5. CONCEPTUAL GROUNDING - Does it explain the physics WHY?\n"
        "6. GRADUATE-LEVEL APPROPRIATENESS - Is the rigor level right?\n\n"
        "Identify the SINGLE issue that would MOST improve the guide's quality. "
        "Prioritize correctness issues first, then completeness, then clarity."
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