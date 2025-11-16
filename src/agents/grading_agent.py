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
    ) -> Dict[str, Any]:

    correct_counts: Dict[str, int] = {}
    scores_by_persona: Dict[str, float] = {}

    question_id = str(gpqa_question.get("id"))
    gold = str(gpqa_question.get("correct_answer", "")).strip().upper()
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
        expert_text = str(gpqa_question.get("expert_explanation") or gpqa_question.get("explanation", ""))
        jr = judge_explanation(expert_text, jt)
        score = int(jr["explanation_score"])  # 1-5 integer
        j_scores.append(max(0.0, min(1.0, score / 5.0)))
        scores_by_persona[persona] = (sum(j_scores) / len(j_scores)) if j_scores else 0.0
        correct_counts[persona] = correct 
    
    overall = sum(scores_by_persona.values()) / len(scores_by_persona) if scores_by_persona else 0.0
    
    return {
        "total_questions": total,
        "correct_counts": correct_counts,
        "scores_by_persona": scores_by_persona,
        "overall_score": overall,
    }


def grade_gpqa_single(
        gpqa_question: Dict[str, Any],
        letter: str,
        one_sentence: str,
    ) -> Dict[str, Any]:
    qid = str(gpqa_question.get("id", ""))
    gold_letter = str(
        gpqa_question.get("correct") or gpqa_question.get("correct_letter", "")
    ).strip().upper()
    is_correct = (letter.strip().upper() == gold_letter) if gold_letter else False

    expert = str(gpqa_question.get("expert_explanation") or gpqa_question.get("explanation", ""))
    jr = judge_explanation(expert, one_sentence or "")
    score = int(jr["explanation_score"])  # 1-5
    sim = max(0.0, min(1.0, score / 5.0))

    return {
        "total_questions": 1,
        "question_id": qid,
        "predicted": letter.strip().upper(),
        "is_correct": bool(is_correct),
        "explanation_one_liner": one_sentence or "",
        "explanation_similarity": sim,
    }


def grading_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gpqa_question = state.get("gpqa_question", {})
    explanation = str(state.get("explanation", ""))
    if not isinstance(gpqa_question, Dict) or not gpqa_question:
        raise ValueError("GPQA question is required in state for grading.")
    # Single-answer mode if present, else legacy multi-persona
    if "single_answer" in state:
        results = grade_gpqa_single(
            gpqa_question,
            str(state.get("single_answer", "")),
            str(state.get("single_explanation", "")),
        )
    else:
        student_answers = state.get("student_answers", {})
        student_justifications = state.get("student_justifications", {})
        results = grade_gpqa(gpqa_question, student_answers, student_justifications)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "gpqa_question": gpqa_question,
        "single_answer": state.get("single_answer"),
        "single_explanation": state.get("single_explanation"),
        "quiz_results": results,
    })
    return {
        "gpqa_question": gpqa_question,
        "single_answer": state.get("single_answer"),
        "single_explanation": state.get("single_explanation"),
        "quiz_results": results,
        "history": history,
    }

