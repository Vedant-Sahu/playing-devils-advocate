from __future__ import annotations

import os
from typing import Any, Dict, List
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm
from src.utils.parsing import _extract_json

try:
    from trulens.core.otel.instrument import instrument  # type: ignore
except Exception:  # pragma: no cover
    def instrument(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


def _load_rubric() -> Dict[str, Any]:
    try:
        data_path = Path(__file__).resolve().parents[2] / "data" / "judge_rubric.json"
        if data_path.exists():
            import json
            return json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "metrics": ["clarity", "correctness", "completeness", "alignment"],
        "scale": {"min": 1, "max": 5},
    }


@instrument()
def judge_explanation(
    gpqa_question: Dict[str, Any], 
    explanation: str, 
    topics: List[str] | None = None
) -> Dict[str, Any]:
    rubric = _load_rubric()
    metrics = rubric.get("metrics", ["clarity", "correctness", "completeness", "alignment"])
    question = gpqa_question['question']
    expert_explanation = gpqa_question["explanation"]
    llm = _llm(temperature=0.0, json_mode=True, role="judge", max_tokens=500)
    metrics_lower = [str(m).strip().lower() for m in metrics]
    metrics_line = ", ".join(metrics_lower)
    sys = SystemMessage(
        content=(
            "You are an impartial LLM judge. Evaluate the explanation on the given metrics. "
            "Use exactly these metric keys: "
            + metrics_line
            + ". Score each metric on a 1-5 integer scale and provide a one-sentence rationale per metric. "
            "Return ONLY valid JSON with shape: {\"scores\": {<metric>: 1-5, ...}, \"rationales\": {<metric>: \"...\"}}. "
            "Do not include any extra metrics or keys."
        )
    )
    topics_line = ", ".join([str(t) for t in topics]) if topics else ""
    hum = HumanMessage(
        content=(
            (f"Question:\n{question}\n\n" if question else "")
            + (f"Topics to cover: {topics_line}\n\n" if topics_line else "")
            + f"Explanation to evaluate:\n{explanation}"
        )
    )
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Judge must return a JSON object.")
    scores_raw = parsed.get("scores")
    rationales_raw = parsed.get("rationales")
    if not isinstance(scores_raw, dict):
        raise ValueError("Judge JSON must include a 'scores' object.")
    if rationales_raw is not None and not isinstance(rationales_raw, dict):
        raise ValueError("Judge 'rationales' must be an object if provided.")
    # Normalize score keys to lowercase for strict matching against required metrics
    scores = {str(k).strip().lower(): v for k, v in scores_raw.items()}
    rationales = {str(k).strip().lower(): v for k, v in (rationales_raw or {}).items()}
    sc: Dict[str, int] = {}
    for m in metrics_lower:
        if m not in scores:
            raise ValueError(f"Missing score for metric '{m}'.")
        try:
            v = int(scores[m])
        except Exception:
            raise ValueError(f"Non-integer score for metric '{m}'.")
        if v < 1 or v > 5:
            raise ValueError(f"Score for metric '{m}' must be in 1–5.")
        sc[m] = v
    overall = sum(sc.values()) / len(sc) if sc else 0.0
    return {"scores": sc, "rationales": {k: str(v) for k, v in rationales.items()}, "overall": overall}


@instrument()
def judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    enabled = os.getenv("ENABLE_JUDGE", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {}
    gpqa_question = state.get("gpqa_question", "")
    explanation = state.get("explanation", "")
    result = judge_explanation(gpqa_question, explanation)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "judge": result,
    })
    return {
        "judge_scores": result.get("scores", {}),
        "judge_rationales": result.get("rationales", {}),
        "judge_overall": result.get("overall", 0.0),
        "history": history,
    }
