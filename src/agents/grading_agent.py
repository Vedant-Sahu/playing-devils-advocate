from __future__ import annotations

from typing import Any, Dict, List

from langchain.schema import HumanMessage, SystemMessage

from .common import _llm, _extract_json, PERSONAS, PERSONA_GUIDELINES
from .judge_agent import judge_explanation


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


def answer_gpqa_for_persona(persona: str, explanation: str, quiz: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    guide = PERSONA_GUIDELINES.get(persona.lower().strip(), "You are a student.")
    llm = _llm(temperature=0.0, json_mode=True, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Answer the multiple-choice question(s) strictly using what is conveyed in the teacher's explanation. "
            + "For each question, select exactly one letter A–D and provide a concise justification of 1–3 sentences. "
            + "Return ONLY JSON of the form {\"answers\": {\"<ID>\": \"A\", ...}, \"justifications\": {\"<ID>\": \"...\", ...}}. "
            + "Use the exact question ID string(s) printed in the quiz as keys (do NOT use 'q1')."
        )
    )
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
            "Teacher explanation (context for answering):\n" + str(explanation) + "\n\n"
            + "Answer the following quiz based only on the explanation above.\n\n"
            + quiz_text
        )
    )
    try:
        resp = llm.invoke([sys, hum])
        raw = resp.content
        parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
        answers = parsed.get("answers") if isinstance(parsed, dict) else None
        justifs = parsed.get("justifications") if isinstance(parsed, dict) else None
        result_answers: Dict[str, str] = {}
        result_justifs: Dict[str, str] = {}
        if isinstance(answers, dict):
            for q in quiz:
                qid = str(q.get("id"))
                if not qid:
                    continue
                letter = str(answers.get(qid, "")).strip().upper()
                if letter in {"A", "B", "C", "D"}:
                    result_answers[qid] = letter
        if isinstance(justifs, dict):
            for q in quiz:
                qid = str(q.get("id"))
                if not qid:
                    continue
                jt = str(justifs.get(qid, "")).strip()
                result_justifs[qid] = jt
        # Single-question tolerance: if model used a different single key, map it to the real ID
        if len(quiz) == 1:
            qid0 = str(quiz[0].get("id"))
            if qid0 and qid0 not in result_answers and isinstance(answers, dict) and len(answers) == 1:
                _k = next(iter(answers.keys()))
                _v = str(answers.get(_k, "")).strip().upper()
                if _v in {"A", "B", "C", "D"}:
                    result_answers[qid0] = _v
            if qid0 and qid0 not in result_justifs and isinstance(justifs, dict) and len(justifs) == 1:
                _k2 = next(iter(justifs.keys()))
                _j = str(justifs.get(_k2, "")).strip()
                if _j:
                    result_justifs[qid0] = _j
        # Validate presence for each quiz id; raise if missing
        for q in quiz:
            qid = str(q.get("id"))
            if not qid:
                raise ValueError("Quiz item missing 'id'.")
            letter = result_answers.get(qid, "")
            if letter not in {"A", "B", "C", "D"}:
                raise ValueError(f"Missing or invalid answer for persona '{persona}' and id '{qid}'.")
            jt = result_justifs.get(qid, "").strip()
            if not jt:
                raise ValueError(f"Missing justification for persona '{persona}' and id '{qid}'.")
        return {"answers": result_answers, "justifications": result_justifs}
    except Exception as e:
        raise RuntimeError(f"Failed to collect answers for persona '{persona}': {e}")


def grade_gpqa(quiz: List[Dict[str, Any]], answers_by_persona: Dict[str, Dict[str, str]], justifs_by_persona: Dict[str, Dict[str, str]], weight_correctness: float = 0.8) -> Dict[str, Any]:
    total = len(quiz) if quiz else 0
    correct_counts: Dict[str, int] = {}
    scores_by_persona: Dict[str, float] = {}
    just_scores_by_persona: Dict[str, float] = {}
    for persona in PERSONAS:
        answers = answers_by_persona.get(persona, {})
        justifs = justifs_by_persona.get(persona, {})
        correct = 0
        j_scores: list[float] = []
        for q in quiz:
            qid = str(q.get("id"))
            gold = str(q.get("correct", "")).strip().upper()
            pred = str(answers.get(qid, "")).strip().upper()
            if qid and pred == gold:
                correct += 1
            jt = str(justifs.get(qid, "")).strip()
            try:
                jr = judge_explanation(q.get("stem", ""), jt)
                overall = float(jr.get("overall", 3.0))
                j_scores.append(max(0.0, min(1.0, overall / 5.0)))
            except Exception:
                j_scores.append(0.5)
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
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    quiz = state.get("gpqa_quiz")
    if not isinstance(quiz, list) or not quiz:
        raise ValueError("gpqa_quiz is required in state for grading.")
    answers_by_persona: Dict[str, Dict[str, str]] = {}
    justifs_by_persona: Dict[str, Dict[str, str]] = {}
    for p in PERSONAS:
        res = answer_gpqa_for_persona(p, explanation, quiz)
        answers_by_persona[p] = res.get("answers", {})
        justifs_by_persona[p] = res.get("justifications", {})
    results = grade_gpqa(quiz, answers_by_persona, justifs_by_persona)
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "quiz": quiz,
        "quiz_answers": answers_by_persona,
        "quiz_justifications": justifs_by_persona,
        "quiz_results": results,
    })
    return {
        "quiz": quiz,
        "quiz_answers": answers_by_persona,
        "quiz_justifications": justifs_by_persona,
        "quiz_results": results,
        "history": history,
    }

