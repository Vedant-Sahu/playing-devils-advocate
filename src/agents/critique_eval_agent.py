from __future__ import annotations

from typing import Any, Dict
import re

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, PERSONAS


def _extract_1_2_3(text: str) -> int:
    m = re.search(r"\b([123])\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    try:
        v = int(str(text).strip())
        return v if v in (1, 2, 3) else 2
    except Exception:
        return 2


def reward_score(persona: str, student_response: str) -> int:
    persona = persona.lower().strip()
    rubric = {
        "advanced": (
            "Constructive = sophisticated follow-up questions and identifying technical gaps/edge cases."
        ),
        "struggling": (
            "Constructive = clearly articulating basic confusions and asking for step-by-step scaffolding."
        ),
        "visual": (
            "Constructive = identifying where visualization would help and describing missing mental models."
        ),
        "practical": (
            "Constructive = requesting or evaluating real-world applications and use-case relevance."
        ),
        "theoretical": (
            "Constructive = requesting formal notation/proofs and pointing out logical/rigor gaps."
        ),
    }.get(persona, "Constructive = specific, actionable feedback aligned to the persona.")

    llm = _llm(temperature=0.0, role="critique_eval")
    sys = SystemMessage(
        content=(
            f"You are the Reward Agent for persona '{persona}'. Rate how constructive the student's "
            f"classroom paragraph is based on this rubric: {rubric} "
            "Return ONLY one of: 1, 2, or 3. "
            "Be strict and use the full scale: typical comments are 1–2; 3 is rare. "
            "1 = vague/generic praise, off-topic, repetitive, or asks for more length/examples without clear purpose. "
            "2 = on-topic with a concrete, relevant ask or clear gap; minimal redundancy. "
            "3 = rare: highly specific, actionable, and efficiency-focused (e.g., clarify a step/term or one targeted example). "
            "If unsure, choose the lower score."
        )
    )
    hum = HumanMessage(content="Student response paragraph:\n" + str(student_response))
    resp = llm.invoke([sys, hum])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _extract_1_2_3(content)


def reward_node(state: Dict[str, Any]) -> Dict[str, Any]:
    responses = state.get("student_responses", {})
    scores = {p: reward_score(p, responses.get(p, {})) for p in PERSONAS}
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": state.get("explanation", ""),
        "student_responses": responses,
        "reward_scores": scores,
    })
    return {"reward_scores": scores, "history": history}
