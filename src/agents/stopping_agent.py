from __future__ import annotations

from typing import Any, Dict, List

from .common import StopConfig, PERSONAS


def stopping_decision(
    scores: Dict[str, float],
    history_scores: List[Dict[str, float]],
    iteration: int,
    config: StopConfig | None = None,
) -> Dict[str, str]:
    cfg = config or StopConfig()

    # Safety cap
    if iteration >= cfg.max_iterations:
        return {"decision": "STOP", "reason": "Max iterations reached."}

    # New 1–3 scale rules:
    # - CONTINUE if any persona has score >= 2
    # - STOP if all scores == 1 (no actionable feedback at all)
    # - STOP if all personas returned empty feedback (handled below)
    # - STOP due to stagnation if no 3s for K consecutive rounds (K=2)
    vals = [int(scores.get(p, 0) or 0) for p in PERSONAS]
    if all(v == 1 for v in vals):
        return {"decision": "STOP", "reason": "All personas scored 1 (no actionable feedback)."}
    count3 = sum(1 for v in vals if v == 3)
    if count3 == 0 and cfg.stagnation_window > 0:
        # Check previous rounds for absence of 3s
        window = min(cfg.stagnation_window, len(history_scores))
        no3_prev = True
        for past in history_scores[-window:]:
            pv = [int(past.get(p, 0) or 0) for p in PERSONAS]
            if any(v == 3 for v in pv):
                no3_prev = False
                break
        if no3_prev and window == cfg.stagnation_window:
            return {"decision": "STOP", "reason": f"Stagnation: no score 3 for {cfg.stagnation_window} consecutive rounds."}

    # Otherwise continue refining
    return {"decision": "CONTINUE", "reason": "Multiple strong or mixed signals present; continue refining."}


def stopper_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scores = state.get("reward_scores", {})
    raw_hist = state.get("history", [])
    # Use only prior rounds that actually recorded reward_scores to avoid counting other history entries
    history_scores = [
        h.get("reward_scores", {})
        for h in raw_hist
        if isinstance(h.get("reward_scores"), dict) and h.get("reward_scores")
    ]
    iteration = int(state.get("iteration", 0))
    cfg = StopConfig(
        threshold=state.get("threshold", 0.7),
        max_iterations=state.get("max_iters", 5),
        stagnation_window=int(state.get("stagnation_window", StopConfig().stagnation_window)),
    )
    responses = state.get("student_responses", {})
    all_empty = False
    if isinstance(responses, dict):
        try:
            all_empty = all(isinstance(responses.get(p, {}), dict) and len(responses.get(p, {})) == 0 for p in PERSONAS)
        except Exception:
            all_empty = False
    if all_empty:
        return {"decision": "STOP", "reason": "All personas returned empty feedback."}
    decision = stopping_decision(scores, history_scores, iteration, cfg)
    return {"decision": decision.get("decision"), "reason": decision.get("reason")}
