from __future__ import annotations
from typing import Any, Dict
from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import _extract_json


def student_respond(persona: str, explanation: str) -> Dict[str, Any]:
    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")
    llm = _llm(temperature=0.0, json_mode=True, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Provide constructive feedback on the teacher's explanation."
            " Return ONLY valid JSON. Keys are optional and must be omitted if empty."
            " Allowed keys: worked, didnt_work, requests, confusions."
            " For worked/didnt_work/requests/confusions: arrays of up to 2 short items."
            " Each non-empty item must reference an exact phrase or sentence index from the explanation."
            " If you have nothing to add, return {}."
        )
    )
    hum = HumanMessage(content=f"Explanation:\n{explanation}")
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Student feedback must be a JSON object.")
    allowed = {"worked","didnt_work","requests","confusions"}
    extra_keys = set(parsed.keys()) - allowed
    if extra_keys:
        raise ValueError(f"Unexpected keys in student feedback: {sorted(extra_keys)}")
    result: Dict[str, Any] = {}
    def _clean_list(name: str) -> None:
        v = parsed.get(name)
        if v is None:
            return
        if not isinstance(v, list):
            raise ValueError(f"{name} must be a list if provided.")
        items = [str(x).strip() for x in v if str(x).strip()]
        if not items:
            return
        result[name] = items[:2]
    for k in ["worked","didnt_work","requests","confusions"]:
        _clean_list(k)
    # Best-effort normalization to a single line/paragraph
    return result


def student_critiques_node(state: Dict[str, Any]) -> Dict[str, Any]:
    explanation = state.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation is required in state for students_node.")
    responses: Dict[str, Any] = {}
    for p in PERSONAS:
        fb = student_respond(p, explanation)
        if not isinstance(fb, dict):
            raise ValueError(f"student_respond must return an object for persona '{p}'.")
        responses[p] = fb
    return {"student_responses": responses}


def student_answers(
        persona: str, 
        explanation: str,
        gpqa_question: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")
    llm = _llm(temperature=0.0, json_mode=True, role="student")
    sys = SystemMessage(
        content=(
            guide
            + " Answer the multiple-choice question(s) strictly using what is conveyed in the teacher's explanation. "
            + "For each question, select exactly one option A–D and provide a concise justification of 1–3 sentences. "
            + "Return ONLY JSON of the form {\"answers\": {\"<ID>\": \"A\", ...}, \"justifications\": {\"<ID>\": \"...\", ...}}. "
            + "Use the exact question ID string(s) printed in the quiz as keys (Do NOT use 'q1')."
        )
    )
    quiz_text_lines: list[str] = []
    quiz_text_lines.append(
        f"ID: {gpqa_question.get('id','')}\n"
        f"Question: {gpqa_question.get('question','')}\n"
        f"Options:\n" + "\n".join(gpqa_question.get('options', [])) + "\n"
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
        question_id = str(gpqa_question.get("id"))
        
        if isinstance(answers, dict):
            if not question_id:
                raise ValueError(f"Missing question ID for persona '{persona}'.")
            letter = str(answers.get(question_id, "")).strip().upper()
            if letter in {"A", "B", "C", "D"}:
                result_answers[question_id] = letter
            else:
                raise ValueError(f"Missing or invalid answer for persona '{persona}' and id '{question_id}'.")
        
        if isinstance(justifs, dict):
            if not question_id:
                raise ValueError(f"Missing question ID for persona '{persona}'.")
            jt = str(justifs.get(question_id, "")).strip()
            if not jt:
                raise ValueError(f"Missing justification for persona '{persona}' and id '{question_id}'.")
            result_justifs[question_id] = jt
        return {"answers": result_answers, "justifications": result_justifs}
    except Exception as e:
        raise RuntimeError(f"Failed to collect answers for persona '{persona}': {e}")


def student_answers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    
    explanation = state.get("explanation")
    gpqa_question = state.get("gpqa_question", [])

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Explanation is required in state for student_answers_node.")
    
    if not gpqa_question or not isinstance(gpqa_question, Dict):
        raise ValueError("gpqa_question must be a non-empty dictionary in state.")
    
    answers_by_persona: Dict[str, Dict[str, str]] = {}
    justifs_by_persona: Dict[str, Dict[str, str]] = {}
        
    for p in PERSONAS:
        res = student_answers(p, explanation, gpqa_question)
        answers_by_persona[p] = res.get("answers", {})
        justifs_by_persona[p] = res.get("justifications", {})

    return {"student_answers": answers_by_persona,
            "student_justifications": justifs_by_persona}
