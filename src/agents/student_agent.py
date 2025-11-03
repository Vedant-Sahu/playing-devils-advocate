from __future__ import annotations

from typing import Any, Dict

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, PERSONAS, PERSONA_GUIDELINES


def student_respond(persona: str, explanation: str) -> str:
    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")
    llm = _llm(temperature=0.0, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Read the teacher's explanation and respond according to your learning style. "
            "Return a SINGLE short paragraph (<=3 sentences), written as a student. "
            "Ask at most 1 specific, relevant question only if a concrete gap remains; otherwise do not ask. "
            "If the explanation feels long or padded, request concision rather than more examples. "
            "Avoid repetition or rephrasing; each sentence must add a new point relevant to the main topic. "
            "If there is nothing to ask, respond exactly 'I understand.' "
            "Do NOT use lists, headings, or markdown. No backticks. "
        )
    )
    hum = HumanMessage(content=f"Explanation:\n{explanation}")
    resp = llm.invoke([sys, hum])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    # Best-effort normalization to a single line/paragraph
    paragraph = " ".join(content.strip().split())
    return paragraph


def students_node(state: Dict[str, Any]) -> Dict[str, Any]:
    explanation = state.get("explanation", "")
    responses = {p: student_respond(p, explanation) for p in PERSONAS}
    return {"student_responses": responses}
