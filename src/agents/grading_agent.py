from __future__ import annotations

from typing import Any, Dict, List

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json, PERSONAS, PERSONA_GUIDELINES


def generate_quiz(question: str, explanation: str, num_questions: int = 4) -> List[Dict[str, Any]]:
    """Generate a short MCQ quiz (3-5 Qs) about the explanation.
    Returns a list of items: {id, stem, options: ["A) ...", ...], correct: "A|B|C|D"}.
    """
    n = max(3, min(5, int(num_questions or 4)))
    llm = _llm(temperature=0.2, json_mode=True, role="grading")
    sys = SystemMessage(
        content=(
            "You are the Grading Agent. Create a concise multiple-choice quiz to assess understanding of the provided explanation. "
            "Requirements: "
            "- Focus only on core concepts covered in the explanation; avoid extraneous topics. "
            "- Produce exactly N questions. "
            "- Each question must have exactly 4 options labeled A–D; exactly one correct. "
            "Return ONLY valid JSON with shape: {\"quiz\":[{\"id\":\"q1\",\"stem\":\"...\",\"options\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],\"correct\":\"A\"}, ...]}"
        )
    )
    hum = HumanMessage(
        content=(
            f"N: {n}\n"
            f"Original question:\n{question}\n\n"
            f"Explanation:\n{explanation}"
        )
    )
    try:
        resp = llm.invoke([sys, hum])
        raw = resp.content
        parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
        quiz = parsed.get("quiz") if isinstance(parsed, dict) else None
        items: List[Dict[str, Any]] = []
        if isinstance(quiz, list):
            for i, q in enumerate(quiz, start=1):
                try:
                    qid = str(q.get("id") or f"q{i}")
                    stem = str(q.get("stem") or "")
                    options = [str(o) for o in (q.get("options") or [])][:4]
                    correct = str(q.get("correct") or "").strip().upper()
                    if stem and len(options) == 4 and correct in {"A", "B", "C", "D"}:
                        items.append({"id": qid, "stem": stem, "options": options, "correct": correct})
                except Exception:
                    continue
        if len(items) >= 3:
            return items[:n]
    except Exception:
        pass

    # Fallback minimal quiz if parsing fails
    fallback: List[Dict[str, Any]] = [
        {
            "id": "q1",
            "stem": "Which option best captures a central point of the explanation?",
            "options": ["A) The core mechanism described", "B) An unrelated concept", "C) A tangential detail", "D) A historical note"],
            "correct": "A",
        },
        {
            "id": "q2",
            "stem": "Which statement aligns with the example provided in the explanation?",
            "options": ["A) Matches the example", "B) Contradicts the example", "C) Irrelevant detail", "D) Ambiguous claim"],
            "correct": "A",
        },
        {
            "id": "q3",
            "stem": "Which choice is consistent with the constraints or assumptions discussed?",
            "options": ["A) Consistent", "B) Violates assumptions", "C) Not discussed", "D) Off-topic"],
            "correct": "A",
        },
    ]
    return fallback[:n]


def answer_quiz_for_persona(persona: str, explanation: str, quiz: List[Dict[str, Any]]) -> Dict[str, str]:
    """Ask a persona to answer the quiz based on the explanation. Returns mapping qid -> letter."""
    guide = PERSONA_GUIDELINES.get(persona.lower().strip(), "You are a student.")
    llm = _llm(temperature=0.0, json_mode=True, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Answer the multiple-choice quiz strictly using what is conveyed in the explanation. "
            + "Return ONLY JSON of the form {\"answers\": {\"q1\": \"A\", ...}} with letters A–D. No commentary."
        )
    )
    condensed_quiz = [
        {"id": q["id"], "stem": q["stem"], "options": q["options"]}
        for q in quiz
    ]
    quiz_text_lines: list[str] = []
    for q in quiz:
        try:
            opts = "\n".join([str(o) for o in q.get("options", [])])
        except Exception:
            opts = ""
        quiz_text_lines.append(
            f"ID: {q.get('id','')}\n"
            f"Stem: {q.get('stem','').strip()}\n"
            f"Options:\n{opts}\n"
        )
    quiz_text = "\n".join(quiz_text_lines)
    hum = HumanMessage(
        content=(
            "Explanation:\n" + str(explanation) + "\n\n" +
            "Answer the following quiz based only on the explanation above.\n\n" +
            quiz_text
        )
    )
    try:
        resp = llm.invoke([sys, hum])
        raw = resp.content
        parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
        answers = parsed.get("answers") if isinstance(parsed, dict) else None
        result: Dict[str, str] = {}
        if isinstance(answers, dict):
            for q in quiz:
                qid = q.get("id")
                if not qid:
                    continue
                letter = str(answers.get(str(qid), "")).strip().upper()
                if letter in {"A", "B", "C", "D"}:
                    result[str(qid)] = letter
        # Ensure an answer for each question (default to A)
        for q in quiz:
            qid = str(q.get("id"))
            if qid and qid not in result:
                result[qid] = "A"
        return result
    except Exception:
        # Default all A if failure
        return {str(q.get("id")): "A" for q in quiz if q.get("id")}


def grade_quiz(quiz: List[Dict[str, Any]], answers_by_persona: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    total = len(quiz) if quiz else 0
    per_persona_scores: Dict[str, float] = {}
    correct_counts: Dict[str, int] = {}
    for persona, answers in answers_by_persona.items():
        correct = 0
        for q in quiz:
            qid = str(q.get("id"))
            correct_letter = str(q.get("correct", "")).strip().upper()
            if qid and answers.get(qid, "").strip().upper() == correct_letter:
                correct += 1
        correct_counts[persona] = correct
        per_persona_scores[persona] = (correct / total) if total > 0 else 0.0
    overall = sum(per_persona_scores.values()) / len(per_persona_scores) if per_persona_scores else 0.0
    return {
        "total_questions": total,
        "correct_counts": correct_counts,
        "scores_by_persona": per_persona_scores,
        "overall_score": overall,
    }


def grading_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    quiz = generate_quiz(question, explanation, num_questions=4)
    answers_by_persona: Dict[str, Dict[str, str]] = {}
    for p in PERSONAS:
        answers_by_persona[p] = answer_quiz_for_persona(p, explanation, quiz)
    results = grade_quiz(quiz, answers_by_persona)
    # Append to history for traceability
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "quiz": quiz,
        "quiz_answers": answers_by_persona,
        "quiz_results": results,
    })
    return {
        "quiz": quiz,
        "quiz_answers": answers_by_persona,
        "quiz_results": results,
        "history": history,
    }

