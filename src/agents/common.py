from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

# Personas used across the system
PERSONAS: List[str] = [
    "advanced",
    "struggling",
    "visual",
    "practical",
    "theoretical",
]

# Persona behavior guidelines
PERSONA_GUIDELINES: Dict[str, str] = {
    "advanced": (
        "You are an Advanced Student. You grasp concepts quickly and prefer deeper technical "
        "details, edge cases, and advanced applications. You are frustrated by oversimplified "
        "explanations."
    ),
    "struggling": (
        "You are a Struggling Student. You need concrete examples and scaffolding. You are "
        "confused by jargon and abstract concepts and require step-by-step breakdowns."
    ),
    "visual": (
        "You are a Visual/Spatial Learner. You need diagrams, spatial metaphors, and visual "
        "descriptions and struggle with purely verbal/mathematical explanations."
    ),
    "practical": (
        "You are a Practical/Applied Learner. You want real-world applications and concrete "
        "use cases and get impatient with pure theory."
    ),
    "theoretical": (
        "You are a Theoretical/Mathematical Learner. You prefer formal definitions and "
        "mathematical rigor and want proofs/derivations where appropriate."
    ),
}


def _llm(temperature: float = 0.2, json_mode: bool = False) -> ChatOpenAI:
    if json_mode:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def _extract_json(text: str) -> Any:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError("Could not extract valid JSON from model output.")


def _extract_float(text: str) -> float:
    m = re.search(r"([01](?:\.\d+)?|0?\.\d+)", text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except Exception:
            pass
    return 0.5


@dataclass
class StopConfig:
    threshold: float = 0.7
    max_iterations: int = 5
    stagnation_window: int = 2
    stagnation_min_improvement: float = 0.02
