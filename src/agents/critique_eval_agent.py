from __future__ import annotations
from typing import Any, Dict
from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import _extract_json


def _judge_feedback(
        persona: str, 
        feedback: Dict[str, Any], 
        question: str | None = None, 
        explanation: str | None = None
    ) -> Dict[str, Any]:
    persona = persona.lower().strip()
    if not isinstance(feedback, dict):
        raise ValueError("feedback must be a dict (structured JSON from student).")
    # Empty feedback -> score 1
    if len(feedback) == 0:
        return {"score": 1, "rationales": {}}
    llm = _llm(temperature=1.0, json_mode=True, role="critique_eval")
    guide = PERSONA_GUIDELINES.get(persona, "You are a student.")
    sys = SystemMessage(
        content=(
            "You are the Feedback Judge. Evaluate a student's structured feedback for improving the explanation, with this persona context: "
            + guide
            + " Use exactly these four criteria as subscores (1–5 integers): specificity (text-referential), actionability (clear, feasible revision), alignment (on-topic to question/explanation and persona), nonredundancy. "
            "Return ONLY JSON: {\"subscores\": {\"specificity\":1-5, \"actionability\":1-5, \"alignment\":1-5, \"nonredundancy\":1-5}, \"rationales\": {same keys -> short strings}, \"final\": 1|2|3}. "
            "Compute final from the average of subscores: 1 if avg < 2.5; 2 if 2.5 <= avg < 4.0; 3 if avg >= 4.0 (3 is rare)."
        )
    )
    payload = {
        "persona": persona,
        "question": question or "",
        "explanation": explanation or "",
        "feedback": feedback,
    }
    hum = HumanMessage(content=str(payload))
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Feedback Judge must return JSON with subscores.")
    subs = parsed.get("subscores")
    if not isinstance(subs, dict):
        raise ValueError("Feedback Judge JSON must include 'subscores'.")
    required = ["specificity", "actionability", "alignment", "nonredundancy"]
    scores5: dict[str, int] = {}
    for k in required:
        if k not in subs:
            raise ValueError(f"Missing subscore '{k}'.")
        try:
            v = int(subs[k])
        except Exception:
            raise ValueError(f"Non-integer subscore for '{k}'.")
        if v < 1 or v > 5:
            raise ValueError(f"Subscore for '{k}' must be 1–5.")
        scores5[k] = v
    avg = sum(scores5.values()) / len(scores5)
    computed = 1 if avg < 2.5 else (3 if avg >= 4.0 else 2)
    r = parsed.get("rationales")
    rationales = r if isinstance(r, dict) else {}
    return {"score": computed, "rationales": {k: str(v) for k, v in rationales.items()}}


def reward_score(persona: str, student_feedback: Dict[str, Any], question: str | None = None, explanation: str | None = None) -> int:
    result = _judge_feedback(persona, student_feedback, question, explanation)
    return int(result.get("score", 1))


def reward_node(state: Dict[str, Any]) -> Dict[str, Any]:
    responses = state.get("student_responses", {})
    if not isinstance(responses, dict):
        raise ValueError("student_responses must be a dict.")
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    scores: Dict[str, int] = {}
    for p in PERSONAS:
        fb = responses.get(p, {})
        if not isinstance(fb, dict):
            raise ValueError(f"Feedback for persona '{p}' must be an object.")
        scores[p] = reward_score(p, fb, question, explanation)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "student_responses": responses,
        "reward_scores": scores,
    })
    return {"reward_scores": scores, "history": history}
