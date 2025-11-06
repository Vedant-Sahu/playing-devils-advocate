from __future__ import annotations

from langchain.schema import HumanMessage, SystemMessage

from src.utils.parsing import _extract_json
from src.config.agent_config import _llm

from trulens.core.otel.instrument import instrument 


@instrument()
def judge_explanation(
    expert_explanation: str,
    student_explanation: str,
) -> Dict[str, Any]:
    if not isinstance(expert_explanation, str) or not expert_explanation.strip():
        raise ValueError("expert_explanation must be a non-empty string.")
    if not isinstance(student_explanation, str):
        raise ValueError("student_explanation must be a string.")
    if not student_explanation.strip():
        return {"explanation_score": 0}

    llm = _llm(temperature=0.0, json_mode=True, role="judge", max_tokens=500)

    sys = SystemMessage(
        content=(
            "You are an impartial LLM judge. Compare the student's explanation to the expert explanation and "
            "rate how similar they are in substance and reasoning. Return ONLY valid JSON of the form "
            "{\"explanation_score\": 1|2|3|4|5} where 1 = very different and 5 = very similar. "
            "Do not include any other keys or any prose."
        )
    )

    hum_content = (
        "Expert Explanation:\n" + expert_explanation.strip() + "\n\n"
        + "Student Explanation:\n" + student_explanation.strip()
    )
    hum = HumanMessage(content=hum_content)

    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Judge must return a JSON object.")

    score = parsed.get("explanation_score")
    try:
        score_int = int(score)
    except Exception:
        raise ValueError("'explanation_score' must be an integer.")
    if score_int < 1 or score_int > 5:
        raise ValueError("'explanation_score' must be within 1–5.")

    return {"explanation_score": score_int}

