from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json


def analyze_topics(
    question: str,
    last_explanation: Optional[str] = None,
    last_feedback_by_persona: Optional[Dict[str, str]] = None,
) -> List[str]:
    llm = _llm(json_mode=True, role="coordinator")
    feedback_summary = ""
    if last_feedback_by_persona:
        parts: List[str] = []
        for p, txt in last_feedback_by_persona.items():
            s = str(txt).strip().replace("\n", " ")
            if len(s) > 160:
                s = s[:160]
            parts.append(f"{p}: {s}")
        feedback_summary = " | ".join(parts)

    sys = SystemMessage(
        content=(
            "You are the Coordinator Agent. Your job is to decide WHAT the Teacher should cover next. "
            "Input: the question, the previous explanation, and the latest student paragraphs. "
            "Output: a refreshed, concise, high-priority topic list (max 10) to guide the next explanation. "
            "Keep topics on-topic and specific; avoid overly broad or off-topic items. "
            "Examples of good topics: 'learning rate tradeoffs', 'mini-batch variance', 'stopping criteria'. "
            "Examples of bad topics: 'deep learning overview', 'transformers', 'all of linear algebra'. "
            "Return only JSON with shape: {\"topics\": [\"...\"]}."
        )
    )
    payload = {
        "question": question,
        "last_explanation": (last_explanation or ""),
        "latest_student_feedback": (feedback_summary or ""),
    }
    hum = HumanMessage(content=str(payload))
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    topics = parsed.get("topics") if isinstance(parsed, dict) else None
    if not isinstance(topics, list):
        return ["overview", "key concepts", "examples", "math/rigor", "applications"]
    return [str(t).strip() for t in topics if str(t).strip()]


def coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    last_expl = None
    last_fb = None
    if state.get("history"):
        last = state["history"][-1]
        last_expl = last.get("explanation")
        # latest student_responses is a dict of persona -> paragraph (str)
        last_fb = last.get("student_responses")
    topics = analyze_topics(state["question"], last_expl, last_fb)
    return {"topics": topics}
