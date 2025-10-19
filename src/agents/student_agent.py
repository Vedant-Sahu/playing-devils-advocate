from __future__ import annotations

from typing import Any, Dict

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, PERSONAS, PERSONA_GUIDELINES


def student_respond(persona: str, explanation: str) -> str:
    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")
    llm = _llm(temperature=0.0)
    sys = SystemMessage(
        content=(
            guide
            + " Read the teacher's explanation and respond according to your learning style. "
            "Return a SINGLE concise paragraph (3-5 sentences) written as a student in a classroom. "
            "Ask 1-2 specific, relevant questions that build on the explanation and would help you understand better. "
            "You may briefly note what was unclear or what worked, but keep it on-topic and constructive. "
            "Be conservative: only mention issues clearly relevant to the main topic; avoid speculation or tangents. "
            "If there is nothing to ask, just say 'I understand.' "
            "Do NOT use lists, headings, or markdown. No backticks. "
            "Examples of good questions: \"Could you show a small numeric example to illustrate the effect of the learning rate?\" "
            "and \"How does mini-batch noise change convergence compared to full-batch updates?\" "
            "Examples of bad (off-topic) questions: \"Explain transformers from scratch\", \"Teach me linear algebra\"."
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
