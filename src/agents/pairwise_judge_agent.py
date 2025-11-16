from __future__ import annotations

from typing import Any, Dict, List

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json


def pairwise_judge(
    question: str,
    explanation_a: str,
    explanation_b: str,
    topics: List[str] | None = None,
    label_a: str = "system",
    label_b: str = "baseline",
) -> Dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(explanation_a, str) or not explanation_a.strip():
        raise ValueError("explanation_a must be a non-empty string.")
    if not isinstance(explanation_b, str) or not explanation_b.strip():
        raise ValueError("explanation_b must be a non-empty string.")
    if topics is not None and not isinstance(topics, list):
        raise ValueError("topics must be a list of strings if provided.")

    llm = _llm(temperature=1.0, json_mode=True, role="pairwise_judge")

    metrics_line = ", ".join(["clarity", "correctness", "completeness", "alignment", "efficiency"])  # efficiency = avoids unnecessary length
    sys = SystemMessage(
        content=(
            "You are the Pairwise Explanation Judge. Compare two explanations of the same question head-to-head. "
            "Evaluate on these metrics: "
            + metrics_line
            + ". Choose the better overall explanation. If quality is indistinguishable, return 'tie'. "
            "Return ONLY valid JSON of the form {\"winner\": \"A\"|\"B\"|\"tie\", \"rationales\": {<metric>: \"...\"}}. "
            "Rationales must be concise and specific to the texts."
        )
    )

    payload = {
        "question": question,
        "topics": topics or [],
        "A": {"label": str(label_a), "text": explanation_a},
        "B": {"label": str(label_b), "text": explanation_b},
    }
    hum = HumanMessage(content=str(payload))

    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Pairwise Judge must return a JSON object.")

    w = parsed.get("winner")
    if not isinstance(w, str):
        raise ValueError("Pairwise Judge JSON must include a 'winner' string.")
    winner = w.strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError("'winner' must be one of 'A', 'B', 'tie'.")

    r = parsed.get("rationales")
    rationales = r if isinstance(r, dict) else {}
    return {"winner": winner if winner != "TIE" else "tie", "rationales": {k: str(v) for k, v in rationales.items()}}
