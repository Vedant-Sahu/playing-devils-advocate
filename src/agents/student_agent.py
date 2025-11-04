from __future__ import annotations

from typing import Any, Dict

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, PERSONAS, PERSONA_GUIDELINES, _extract_json


def student_respond(persona: str, explanation: str) -> Dict[str, Any]:
    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")
    llm = _llm(temperature=0.0, json_mode=True, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Provide constructive feedback on the teacher's explanation."
            " Return ONLY valid JSON. Keys are optional and must be omitted if empty."
            " Allowed keys: worked, didnt_work, requests, confusions."
            " For worked/didnt_work/requests/confusions: arrays of up to 2 short items."
            " Each non-empty item must reference an exact phrase or sentence index from the explanation."
            " If you have nothing to add, return {}."
        )
    )
    hum = HumanMessage(content=f"Explanation:\n{explanation}")
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Student feedback must be a JSON object.")
    allowed = {"worked","didnt_work","requests","confusions"}
    extra_keys = set(parsed.keys()) - allowed
    if extra_keys:
        raise ValueError(f"Unexpected keys in student feedback: {sorted(extra_keys)}")
    result: Dict[str, Any] = {}
    def _clean_list(name: str) -> None:
        v = parsed.get(name)
        if v is None:
            return
        if not isinstance(v, list):
            raise ValueError(f"{name} must be a list if provided.")
        items = [str(x).strip() for x in v if str(x).strip()]
        if not items:
            return
        result[name] = items[:2]
    for k in ["worked","didnt_work","requests","confusions"]:
        _clean_list(k)
    # Best-effort normalization to a single line/paragraph
    return result


def students_node(state: Dict[str, Any]) -> Dict[str, Any]:
    explanation = state.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation is required in state for students_node.")
    responses: Dict[str, Any] = {}
    for p in PERSONAS:
        fb = student_respond(p, explanation)
        if not isinstance(fb, dict):
            raise ValueError(f"student_respond must return an object for persona '{p}'.")
        responses[p] = fb
    return {"student_responses": responses}
