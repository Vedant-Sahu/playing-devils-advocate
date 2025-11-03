from __future__ import annotations

import os
from typing import Any, Dict, List
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json

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
def judge_explanation(question: str, explanation: str, topics: List[str] | None = None) -> Dict[str, Any]:
    rubric = _load_rubric()
    metrics = rubric.get("metrics", ["clarity", "correctness", "completeness", "alignment"])
    llm = _llm(temperature=0.0, json_mode=True, role="judge")
    sys = SystemMessage(
        content=(
            "You are an impartial LLM judge. Evaluate the explanation on the given metrics. "
            "Score each metric on a 1-5 integer scale and provide a one-sentence rationale per metric. "
            "Return ONLY valid JSON with shape: {\"scores\": {<metric>: 1-5, ...}, \"rationales\": {<metric>: \"...\"}}."
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
    try:
        resp = llm.invoke([sys, hum])
        raw = resp.content
        parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
        scores = parsed.get("scores") if isinstance(parsed, dict) else None
        rationales = parsed.get("rationales") if isinstance(parsed, dict) else None
        if not isinstance(scores, dict):
            scores = {}
        if not isinstance(rationales, dict):
            rationales = {}
        sc: Dict[str, int] = {}
        for m in metrics:
            try:
                v = int(scores.get(m, 3))
            except Exception:
                v = 3
            v = max(1, min(5, v))
            sc[m] = v
        overall = sum(sc.values()) / len(sc) if sc else 0.0
        return {"scores": sc, "rationales": {k: str(v) for k, v in rationales.items()}, "overall": overall}
    except Exception:
        default = {m: 3 for m in metrics}
        return {"scores": default, "rationales": {m: "default" for m in metrics}, "overall": 3.0}


@instrument()
def judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    enabled = os.getenv("ENABLE_JUDGE", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {}
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    topics = state.get("topics") or []
    result = judge_explanation(question, explanation, topics)
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
