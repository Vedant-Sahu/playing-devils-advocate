from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json


def analyze_topics(
    question: str,
    last_explanation: Optional[str] = None,
    last_feedback_by_persona: Optional[Dict[str, Any]] = None,
    prev_topics: Optional[List[str]] = None,
) -> List[str]:
    llm = _llm(json_mode=True, role="coordinator")
    # Build ranked signals from structured feedback: requests > didnt_work > confusions
    signals_counts: Dict[str, int] = {}
    if last_feedback_by_persona is not None:
        if not isinstance(last_feedback_by_persona, dict):
            raise ValueError("last_feedback_by_persona must be a dict if provided.")
        for p, fb in last_feedback_by_persona.items():
            # Backward compatibility: accept string paragraphs but prefer dicts
            if isinstance(fb, dict):
                for k in ("requests", "didnt_work", "confusions"):
                    items = fb.get(k)
                    if isinstance(items, list):
                        for it in items:
                            s = str(it).strip()
                            if s:
                                # Truncate to keep compact
                                if len(s) > 200:
                                    s = s[:200]
                                signals_counts[s] = signals_counts.get(s, 0) + 1
            elif isinstance(fb, str):
                s = fb.strip().replace("\n", " ")
                if s:
                    if len(s) > 200:
                        s = s[:200]
                    signals_counts[s] = signals_counts.get(s, 0) + 1
            else:
                raise ValueError("Each persona feedback must be a dict or string.")

    if not signals_counts:
        # No actionable signals; if previous topics exist, keep them unchanged
        if prev_topics and isinstance(prev_topics, list) and any(str(t).strip() for t in prev_topics):
            return [str(t).strip() for t in prev_topics if str(t).strip()][:10]
        # Cold start: derive topics from the question alone on the first iteration (strict JSON)
        if isinstance(question, str) and question.strip():
            sys = SystemMessage(
                content=(
                    "You are the Coordinator Agent. First-iteration cold start. "
                    "Given ONLY the question, propose up to 10 specific, on-topic topics to guide the Teacher's first explanation. "
                    "Avoid overly broad items; keep them actionable and aligned to the question. "
                    "Return ONLY JSON: {\"topics\": [\"...\"]}."
                )
            )
            payload = {"question": question.strip()}
            hum = HumanMessage(content=str(payload))
            resp = llm.invoke([sys, hum])
            raw = resp.content
            parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
            if not isinstance(parsed, dict) or "topics" not in parsed:
                raise ValueError("Coordinator (cold start) must return JSON with a 'topics' list.")
            topics = parsed.get("topics")
            if not isinstance(topics, list):
                raise ValueError("Coordinator (cold start) 'topics' must be a list of strings.")
            cleaned = []
            seen = set()
            for t in topics:
                s = str(t).strip()
                if not s:
                    continue
                if s not in seen:
                    seen.add(s)
                    cleaned.append(s)
                if len(cleaned) >= 10:
                    break
            if not cleaned:
                raise ValueError("Coordinator (cold start) returned an empty topics list.")
            return cleaned
        # Otherwise, fail fast (no silent fallbacks)
        raise ValueError("No actionable feedback signals and no previous topics provided.")

    ranked = sorted(signals_counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
    top_signals = [f"{t} (x{c})" for t, c in ranked[:10]]

    sys = SystemMessage(
        content=(
            "You are the Coordinator Agent. Decide WHAT the Teacher should cover next. "
            "Input: the question, previous explanation, and ranked student feedback signals. "
            "Output ONLY JSON: {\"topics\": [\"...\"]} with up to 10 specific, on-topic items. "
            "Avoid broad or off-topic items. Good examples: 'learning rate tradeoffs', 'mini-batch variance', 'stopping criteria'."
        )
    )
    payload = {
        "question": question,
        "last_explanation": (last_explanation or ""),
        "ranked_feedback_signals": top_signals,
        "previous_topics": (prev_topics or []),
    }
    hum = HumanMessage(content=str(payload))
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict) or "topics" not in parsed:
        raise ValueError("Coordinator must return JSON with a 'topics' list.")
    topics = parsed.get("topics")
    if not isinstance(topics, list):
        raise ValueError("'topics' must be a list of strings.")
    cleaned = []
    seen = set()
    for t in topics:
        s = str(t).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
        if len(cleaned) >= 10:
            break
    if not cleaned:
        raise ValueError("Coordinator returned an empty topics list.")
    return cleaned


def coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    last_expl = None
    last_fb = None
    prev_topics = None
    if state.get("history"):
        last = state["history"][-1]
        last_expl = last.get("explanation")
        last_fb = last.get("student_responses")
        prev_topics = last.get("topics") or state.get("topics")
    else:
        prev_topics = state.get("topics")
    topics = analyze_topics(state["question"], last_expl, last_fb, prev_topics)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": last_expl,
        "student_responses": last_fb,
        "topics": topics,
    })
    return {"topics": topics, "history": history}
