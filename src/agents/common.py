from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List
import os
from dotenv import load_dotenv
from pathlib import Path

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

load_dotenv()


def _model_for_role(role: str | None) -> str:
    default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    if role:
        key = {
            "teacher": "TEACHER_MODEL",
            "grading": "GRADING_MODEL",
            "coordinator": "COORDINATOR_MODEL",
            "student": "STUDENT_MODEL",
            "critique_eval": "CRITIQUE_EVAL_MODEL",
        }.get(role.lower().strip())
        if key:
            v = os.getenv(key)
            if v:
                return v
    return default_model


def _llm(temperature: float = 0.2, json_mode: bool = False, role: str | None = None) -> ChatOpenAI:
    model = _model_for_role(role)
    if json_mode:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return ChatOpenAI(model=model, temperature=temperature)


# Attempt to load personas and class distribution from data/student_profiles.json
def _load_personas_from_json() -> tuple[list[str] | None, dict[str, str] | None, dict[str, float] | None]:
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


_p, _g, _dist = _load_personas_from_json()
if _p:
    PERSONAS = _p
if _g:
    PERSONA_GUIDELINES.update({k: v for k, v in _g.items() if v})
# Optional export of class distribution for future weighting use
CLASS_DISTRIBUTION: Dict[str, float] = (
    {p: 1.0 / len(PERSONAS) for p in PERSONAS} if not _dist else {str(k): float(v) for k, v in _dist.items() if str(k) in PERSONAS}
)


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
