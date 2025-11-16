"""
Teacher Agent - Generates educational explanations in baseline or adaptive mode.

Baseline mode: Zero-shot explanation without examples or feedback
Adaptive mode: Few-shot with iterative refinement based on student feedback
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm

try:
    from trulens.core.otel.instrument import instrument  # type: ignore
except Exception:  # pragma: no cover
    def instrument(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


def _build_teacher_prompt(
    mode: Literal["baseline", "adaptive"],
    question: str,
    student_feedback: Optional[Dict[str, Dict[str, Any]]] = None,
    word_cap: int = 180,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Build system and human messages for the teacher agent based on mode.
    
    Args:
        mode: "baseline" for zero-shot, "adaptive" for few-shot with refinement
        question: The question to explain
        student_feedback: Feedback from student personas (adaptive mode only)
        word_cap: Maximum word count for explanation
        
    Returns:
        Tuple of (system_message, human_message)
    """
    if mode == "baseline":
        # Zero-shot: No examples, no feedback, pure explanation
        sys = SystemMessage(
            content=(
                "You are an expert teacher teaching undergraduate Physics students with "
                "varying skills and backgrounds. Your goal is to provide a clear, accurate "
                "explanation that helps students understand the given question and its "
                "underlying concepts. "
                f"Provide a concise explanation (maximum {word_cap} words) that includes: "
                "(1) the key concepts needed to understand the question, "
                "(2) a step-by-step explanation of the core mechanism, "
                "(3) a brief example if helpful, "
                "(4) any important definitions or formulas. "
                "Write in clear prose without headers or bullet points. "
                "CRITICAL: DO NOT directly reference the given question or reveal the correct "
                "answer. Teach the underlying concepts generically so students can apply them "
                "to solve the problem independently."
            )
        )
        hum = HumanMessage(
            content=f"Question: {question}\n\nProvide a clear explanation."
        )
        
    else:  
        # Few-shot with refinement: Include example and process feedback
        feedback_summary = _process_feedback(student_feedback)
        has_feedback = bool(feedback_summary)
        
        sys = SystemMessage(
            content=(
                "You are the Teacher Agent in an adaptive learning system. You are teaching "
                "undergraduate Physics students with varying skills and backgrounds. "
                "Role: Produce a clear, self-contained explanation that helps students understand " 
                "the given question and its underlying concepts. "
                "On first iteration, create a well-structured explanation covering key concepts. "
                "On later rounds, revise based on student feedback. Prefer tightening, clarifying, "
                "or replacing over adding new material. "
                "If feedback is empty (all personas returned {}), make no change. "
                "\n\n"
                "Output format: Single block of prose (no headings, no bullet lists). "
                "Aim for concise, structured prose (6-10 sentences). Include: "
                "(1) short intuitive orientation, "
                "(2) core mechanism step-by-step with a tiny numeric example (at most one), "
                "(3) brief visual/spatial analogy if helpful, " 
                "(4) short rigorous note (key definitions/equations) where appropriate. "
                "Each sentence should add new information. "
                f"Limit explanation to no more than {word_cap} words. "
                "CRITICAL: DO NOT directly reference the given question or reveal the correct "
                "answer. If you include any examples in your explanation, do not use any "
                "information directly mentioned in the problem. Teach the underlying concepts "
                "generically so students can apply them to solve the problem independently. "
                "Example of the explanation style:\n"
                "Question: An electron is at rest (not moving). A relativistic positron is moving " 
                "horizontally from the left with a constant speed.\nAfter hitting the electron, " 
                "both annihilate producing 2 photons.\n\nThe direction of one of the  photons is " 
                "in the upper-right direction. The angle between this direction and the horizontal " 
                "line/axis is 60 degrees. The photon energy is 0.613 MeV (1.2 times the rest mass " 
                "of an electron). \n\nWhat was the speed of the positron (expresses as a fraction " 
                "of the speed of light c):\n"
                "Explanation: When matter and antimatter collide and annihilate, they convert their "
                "mass-energy into photons, conserving both energy and momentum. For relativistic "
                "particles moving at speeds comparable to light, we use E² = (pc)² + (mc²)² where E "
                "is total energy, p is momentum, m is rest mass, and c is light speed. For photons "
                "with zero rest mass, E = pc. The Lorentz factor γ = 1/√(1 - v²/c²) relates particle "
                "speed to energy: E = γmc² and momentum p = γmv. Apply conservation laws: total " 
                "initial energy equals sum of photon energies; initial momentum vector equals vector "
                "sum of photon momenta. Break momentum into horizontal and vertical components. The "
                "photon angle and energy reveal the initial particle's momentum and thus its Lorentz "
                "factor. From γ, extract speed using v/c = √(1 - 1/γ²). This framework applies broadly "
                "to two-body decay and annihilation processes. "
                "Use this explanation as a guide for the question provided by the user."
            )
        )
        
        # Build human message with optional feedback
        if has_feedback:
            fb_text = f"\nStudent feedback to address (ranked by frequency):\n{feedback_summary}"
        else:
            fb_text = "\nNo actionable feedback this round." if student_feedback is not None else ""
        
        hum = HumanMessage(
            content=f"Question: {question}{fb_text}\n\nProvide the explanation."
        )
    
    return sys, hum


def _process_feedback(
    student_feedback: Optional[Dict[str, Dict[str, Any]]]
) -> str:
    """
    Process and rank student feedback by frequency.
    
    Args:
        student_feedback: Dict mapping persona names to feedback dicts
        
    Returns:
        Formatted string with top feedback items, or empty string if none
    """
    if not student_feedback or not isinstance(student_feedback, dict):
        return ""
    
    counts: Dict[str, int] = {}
    
    for persona, fb in student_feedback.items():
        if not isinstance(fb, dict):
            continue
            
        # Aggregate feedback from all categories
        for category in ("requests", "didnt_work", "confusions"):
            items = fb.get(category)
            if isinstance(items, list) and items:
                for item in items:
                    s = str(item).strip()
                    if not s:
                        continue
                    # Truncate very long feedback
                    if len(s) > 200:
                        s = s[:200]
                    counts[s] = counts.get(s, 0) + 1
    
    if not counts:
        return ""
    
    # Rank by frequency, then by length (shorter first)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
    
    # Take top 6 items
    top_items = [f"• {text} (x{count})" for text, count in ranked[:6]]
    
    return "\n".join(top_items)


@instrument()
def teacher_explain(
    question: str,
    mode: Literal["baseline", "adaptive"] = "adaptive",
    student_feedback: Optional[Dict[str, Dict[str, Any]]] = None,
    word_cap: int = 180,
    max_tokens: int = 500
) -> str:
    """
    Generate an explanation for the given question.
    
    Args:
        question: The question to explain
        mode: "baseline" for zero-shot or "adaptive" for iterative refinement
        student_feedback: Feedback from student personas (adaptive mode only)
        word_cap: Maximum word count for explanation
        
    Returns:
        Generated explanation text
    """
    llm = _llm(role="teacher", max_tokens=max_tokens)

    # Build prompts based on mode
    sys, hum = _build_teacher_prompt(mode, question, student_feedback, word_cap)

    # Generate explanation
    resp = llm.invoke([sys, hum])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    
    # Clean and truncate
    text = " ".join(content.strip().split())
    words = text.split()
    if len(words) > word_cap:
        text = " ".join(words[:word_cap])
    
    return text


def adaptive_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for adaptive refinement graph.
    
    Uses adaptive mode with student feedback for iterative improvement.
    Extracts question from gpqa_question in state.
    """
    iteration = int(state.get("iteration", 0)) + 1

    # Extract question from gpqa_question
    gpqa_question = state.get("gpqa_question", [])
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    
    # Get feedback from previous iteration
    last_feedback = None
    history = state.get("history", [])
    if history:
        last_feedback = history[-1].get("student_responses")
    
    # Generate explanation in adaptive mode
    explanation = teacher_explain(
        question=question,
        mode="adaptive",
        student_feedback=last_feedback,
        word_cap=180
    )
    
    return {"explanation": explanation, "iteration": iteration}


def baseline_teacher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Teacher node for baseline graph.
    
    Uses zero-shot mode without examples or feedback.
    Extracts question from gpqa_question in state.
    """
    # Extract question from gpqa_question
    gpqa_question = state.get("gpqa_question", [])
    if not gpqa_question:
        raise ValueError("gpqa_question not found in state")
    question = gpqa_question.get("question", "")
    
    # Generate explanation in baseline (zero-shot) mode
    explanation = teacher_explain(
        question=question,
        mode="baseline",
        student_feedback=None,
        word_cap=180
    )
    
    return {"explanation": explanation, "iteration": 1}