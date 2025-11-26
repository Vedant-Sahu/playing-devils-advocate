"""Evaluation metrics for DSPy physics optimization."""

from __future__ import annotations

from typing import Any, Dict

import dspy


def _extract_letter(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip().upper()
    if not text:
        return None
    letter = text[0]
    if letter in {"A", "B", "C", "D"}:
        return letter
    return None


def multiple_choice_accuracy(
    example: dspy.Example | Dict[str, Any],
    prediction: Dict[str, Any] | Any,
    trace: Any = None,
) -> float:
    """Return 1.0 if the predicted letter matches the ground-truth letter, else 0.0."""

    if isinstance(example, dict):
        true_letter = _extract_letter(example.get("correct_letter"))
    else:
        true_letter = _extract_letter(getattr(example, "correct_letter", None))

    if isinstance(prediction, dict):
        pred_letter = _extract_letter(
            prediction.get("final_answer_letter") or prediction.get("final_answer")
        )
    else:
        pred_letter = _extract_letter(getattr(prediction, "final_answer_letter", None))

    if true_letter is None or pred_letter is None:
        return 0.0
    return 1.0 if true_letter == pred_letter else 0.0


def gepa_multiple_choice_metric(
    example: dspy.Example | Dict[str, Any],
    prediction: Dict[str, Any] | Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """GEPA-compatible metric wrapper around multiple_choice_accuracy.

    GEPA expects a metric with signature (gold, pred, trace, pred_name, pred_trace)
    that returns a dspy.Prediction containing at least a numeric `score` field
    and optional `feedback`. Here we reuse multiple_choice_accuracy as the
    underlying score and attach a minimal textual feedback string.
    """

    score = multiple_choice_accuracy(example, prediction, trace)

    # Minimal feedback: note whether the answer was correct or not. This gives
    # GEPA some text to reflect on without changing the task semantics.
    if isinstance(example, dict):
        gold_letter = _extract_letter(example.get("correct_letter"))
    else:
        gold_letter = _extract_letter(getattr(example, "correct_letter", None))

    if isinstance(prediction, dict):
        pred_letter = _extract_letter(
            prediction.get("final_answer_letter") or prediction.get("final_answer")
        )
    else:
        pred_letter = _extract_letter(getattr(prediction, "final_answer_letter", None))

    if gold_letter is None:
        feedback = "Missing gold answer; treating as incorrect."
    elif pred_letter is None:
        feedback = (
            f"Prediction did not include a valid answer letter. Gold was '{gold_letter}'."
        )
    elif score == 1.0:
        feedback = f"Correct multiple-choice answer: '{gold_letter}'."
    else:
        feedback = f"Incorrect multiple-choice answer. Gold is '{gold_letter}', predicted '{pred_letter}'."

    return dspy.Prediction(score=score, feedback=feedback)
