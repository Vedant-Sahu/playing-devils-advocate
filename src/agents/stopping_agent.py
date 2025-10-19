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
    # - STOP if no persona gets a 3
    # - STOP if exactly one persona gets a 3 AND none get a 2
    vals = [int(scores.get(p, 0) or 0) for p in PERSONAS]
    count3 = sum(1 for v in vals if v == 3)
    count2 = sum(1 for v in vals if v == 2)

    if count3 == 0:
        return {"decision": "STOP", "reason": "No persona rated 3 (highly constructive)."}

    if count3 == 1 and count2 == 0:
        return {"decision": "STOP", "reason": "Only one persona at 3 and none at 2."}

    # Otherwise continue refining
    return {"decision": "CONTINUE", "reason": "Multiple strong or mixed signals present; continue refining."}


def stopper_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scores = state.get("reward_scores", {})
    history_scores = [h.get("reward_scores", {}) for h in state.get("history", [])]
    iteration = int(state.get("iteration", 0))
    cfg = StopConfig(threshold=state.get("threshold", 0.7), max_iterations=state.get("max_iters", 5))
    decision = stopping_decision(scores, history_scores, iteration, cfg)
    return {"decision": decision.get("decision"), "reason": decision.get("reason")}
