from __future__ import annotations
from typing import Any, Dict, List
from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import  _extract_json
from src.agents.judge_agent import judge_explanation


def grade_gpqa(
        gpqa_question: Dict[str, Any], 
        student_answers: Dict[str, Dict[str, str]], 
        student_justifications: Dict[str, Dict[str, str]], 
        weight_correctness: float = 0.8
    ) -> Dict[str, Any]:

    correct_counts: Dict[str, int] = {}
    scores_by_persona: Dict[str, float] = {}
    just_scores_by_persona: Dict[str, float] = {}

    question_id = str(gpqa_question.get("id"))
    gold = str(gpqa_question.get("correct", "")).strip().upper()
    total = 1 
    
    for persona in PERSONAS:
        answers = student_answers.get(persona, {})
        justifs = student_justifications.get(persona, {})
        correct = 0
        j_scores: list[float] = []

        pred = str(answers.get(question_id, "")).strip().upper()    
        if question_id and pred == gold:
            correct += 1

        jt = str(justifs.get(question_id, "")).strip()
        jr = judge_explanation(gpqa_question.get("question", ""), jt)
        overall = float(jr["overall"])  # judge_explanation guarantees 'overall'

        j_scores.append(max(0.0, min(1.0, overall / 5.0)))
        correct_frac = (correct / total) if total > 0 else 0.0
        just_mean = (sum(j_scores) / len(j_scores)) if j_scores else 0.0
        score = weight_correctness * correct_frac + (1.0 - weight_correctness) * just_mean
        correct_counts[persona] = correct
        scores_by_persona[persona] = score
        just_scores_by_persona[persona] = just_mean
    
    overall = sum(scores_by_persona.values()) / len(scores_by_persona) if scores_by_persona else 0.0
    
    return {
        "total_questions": total,
        "correct_counts": correct_counts,
        "scores_by_persona": scores_by_persona,
        "justification_scores_by_persona": just_scores_by_persona,
        "overall_score": overall,
    }


def grading_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gpqa_question = state.get("gpqa_question", {})
    explanation = str(state.get("explanation", ""))
    if not isinstance(gpqa_question, Dict) or not gpqa_question:
        raise ValueError("GPQA question is required in state for grading.")
    student_answers = state.get("student_answers", {})
    student_justifications = state.get("student_justifications", {})
    results = grade_gpqa(gpqa_question, student_answers, student_justifications)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "gpqa_question": gpqa_question,
        "student_answers": student_answers,
        "student_justifications": student_justifications,
        "quiz_results": results,
    })
    return {
        "gpqa_question": gpqa_question,
        "student_answers": student_answers,
        "student_justifications": student_justifications,
        "quiz_results": results,
        "history": history,
    }

