from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm

try:
    from trulens.core.otel.instrument import instrument  # type: ignore
except Exception:  # pragma: no cover
    def instrument(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


@instrument()
def teacher_explain(
    question: str,
    topics: List[str],
    student_feedback_by_persona: Optional[Dict[str, Dict[str, Any]]] = None,
    max_tokens: int = 700,
    word_cap: int = 180,
) -> str:
    llm = _llm(role="teacher")
    feedback_summary = ""
    has_any_feedback = False
    if student_feedback_by_persona is not None:
        if not isinstance(student_feedback_by_persona, dict):
            raise ValueError("student_feedback_by_persona must be a dict.")
        counts: Dict[str, int] = {}
        for p, fb in student_feedback_by_persona.items():
            if not isinstance(fb, dict):
                raise ValueError("Each persona's feedback must be an object.")
            for k in ("requests", "didnt_work", "confusions"):
                items = fb.get(k)
                if isinstance(items, list) and items:
                    has_any_feedback = True
                    for it in items:
                        s = str(it).strip()
                        if not s:
                            continue
                        if len(s) > 200:
                            s = s[:200]
                        counts[s] = counts.get(s, 0) + 1
        if counts:
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
            top = [f"{t} (x{c})" for t, c in ranked[:6]]
            feedback_summary = " | ".join(top)

    sys = SystemMessage(
        content=(
            "You are the Teacher Agent. Role: produce a clear, self-contained explanation that "
            "answers the question while covering ALL required topics. On later rounds, revise minimally to "
            "address the latest student feedback and the refreshed topic list; prefer tightening or replacing over adding new material. "
            "If the latest round has no actionable feedback (all personas returned {}), make no change or at most one micro-clarification. "
            "Output: a single block of prose (no headings, no bullet lists). Aim for concise, structured prose (6-10 sentences). "
            "Include: (1) a short intuitive orientation, (2) the core mechanism step-by-step with a tiny numeric example (at most one), "
            "(3) a brief visual/spatial analogy if helpful, and (4) a short rigorous note (key definitions/equations) where appropriate. "
            "Do not introduce new topics beyond the provided list. Avoid padding, restatements, and multiple examples. Each sentence should add new information. Limit the explanation to no more than "
            + str(word_cap)
            + " words."
        )
    )
    plan = (
        "Topics to cover: " + ", ".join(topics)
        if topics
        else "Topics to cover: overview, key concepts, examples, math/rigor, applications"
    )
    fb = (
        f"Top feedback to address (ranked): {feedback_summary}"
        if feedback_summary
        else ("No actionable feedback this round." if student_feedback_by_persona is not None and not has_any_feedback else "")
    )
    hum = HumanMessage(content=f"Question: {question}\n{plan}\n{fb}\nReturn only the explanation text.")
    resp = llm.invoke([sys, hum])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = " ".join(content.strip().split())
    words = text.split()
    if len(words) > word_cap:
        text = " ".join(words[:word_cap])
    return text


def teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    iteration = int(state.get("iteration", 0)) + 1
    last_feedback = None
    if state.get("history"):
        last_feedback = state["history"][-1].get("student_responses")
    explanation = teacher_explain(state["question"], state.get("topics", []), last_feedback)
    return {"explanation": explanation, "iteration": iteration}
