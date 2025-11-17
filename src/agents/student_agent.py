from __future__ import annotations
from typing import Any, Dict
from langchain.schema import HumanMessage, SystemMessage

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import _extract_json, extract_letter_a_to_d, extract_one_sentence


def student_respond(
        persona: str,
        explanation: str,
        score_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:

    key = persona.lower().strip()
    guide = PERSONA_GUIDELINES.get(key, "You are a student.")

    # Build performance context
    performance_context = ""
    if score_history and len(score_history) > 0:
        latest = score_history[-1]
        performance_context = f"""
            YOUR PERFORMANCE HISTORY:
            - Last round score: {latest['score']:.1f}/10 (ranked #{latest['rank']}/5 students)
            - Your previous issue: "{latest['issue'][:80]}..."

            SCORING SYSTEM:
            - Base points: Judge evaluates severity of your issue (1-3 points)
            - Bonus points: Uniqueness (up to 3 points for issues others don't find)
            - Top 3 students influence the teacher's revision
            - Bottom 2 students are ignored

            STRATEGY TIPS:
            - Find genuinely critical flaws (high severity)
            - Look for issues other students might miss (uniqueness bonus)
            - Be specific and cite exact quotes
            - If you ranked low last round, try a different angle
            """
    else:
        performance_context = """
            SCORING SYSTEM:
            - Independent judge evaluates how critical your issue is (1-3 severity points)
            - You earn bonus points for finding unique issues others miss (up to 3 points)
            - Only top 3 students' feedback reaches the teacher
            - Competition is fierce - make your critique count!
            """
    
    llm = _llm(temperature=1.0, json_mode=True, role="student", max_tokens=5000)
    
    sys = SystemMessage(
        content=(
            guide
            + "\n\nYour task: Identify the SINGLE MOST IMPORTANT issue with this explanation.\n"
            " Focus on problems that would confuse students or create misconceptions.\n\n"
            " Return a JSON object with:\n"
            " - 'issue': Brief description of the main problem (max 100 words)\n"
            " - 'quote': Exact phrase from explanation that demonstrates the issue\n\n"
            " If the explanation is genuinely good with no significant issues, return:\n"
            " {'issue': null, 'severity': 0, 'quote': null}\n\n"
            " DO NOT rate severity yourself - independent judges will evaluate that.\n"
            " DO NOT provide generic praise or multiple small issues.\n"
            " Focus on finding the most important issue you can identify.\n"
        )
    )
    
    hum = HumanMessage(content=f"Explanation:\n{explanation}")
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    # Validation
    if not isinstance(parsed, dict):
        raise ValueError("Student feedback must be a JSON object.")
    
    required_keys = {"issue", "quote"}
    if set(parsed.keys()) != required_keys:
        raise ValueError(f"Expected keys {required_keys}, got {set(parsed.keys())}")
    
    # Return normalized format
    return {
        "persona": persona,
        "issue": parsed.get("issue"),
        "quote": parsed.get("quote")
    }


def student_critiques_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """All students provide their top issue"""
    
    explanation = state.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation is required in state for students_node.")
    
    # Get score history for each student
    score_history = state.get("student_score_history", {})

    responses: Dict[str, Any] = {}
    
    for p in PERSONAS:
        persona_history = score_history.get(p, [])
        fb = student_respond(p, explanation, persona_history)
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
    llm = _llm(temperature=0.0, json_mode=True, role="student", max_tokens=2000)
    sys = SystemMessage(
        content=(
            guide
            + " Answer the multiple-choice question strictly using the information in the teacher's explanation. "
            + "Be careful while solving the question and make sure to check your math and reasoning. "
            + "For each question, select exactly one option A–D and provide a concise justification of 1–2 sentences. "
            + "Do NOT exceed more than 100 words for your justification. "
            + "Do NOT include any extra text, explanation, or commentary. "
            + "Return ONLY with a short JSON of the form {\"answers\": {\"<ID>\": \"A\", ...}, \"justifications\": {\"<ID>\": \"...\", ...}}. "
            + "Use the exact question ID as the key (Do NOT use 'q1')."
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


def single_answer(explanation: str, gpqa_question: Dict[str, Any]) -> Dict[str, str]:
    llm = _llm(temperature=1.0, json_mode=True, role="answerer")
    sys = SystemMessage(content=(
        "Return ONLY valid JSON with keys 'answer' and 'explanation'. "
        "Constraints: 'answer' must be exactly one of ['A','B','C','D'] (uppercase). "
        "'explanation' must be a single sentence."
    ))

    qid = str(gpqa_question.get("id", ""))
    qstem = str(gpqa_question.get("question", ""))
    options_list = [str(x) for x in (gpqa_question.get("options", []) or [])]
    options_text = "\n".join(options_list)

    hum = HumanMessage(content=(
        "Teacher explanation (context):\n" + str(explanation) + "\n\n"
        + "Question: " + qstem + "\n"
        + "Options:\n" + options_text + "\n"
        + "Respond as JSON only."
    ))

    resp = llm.invoke([sys, hum])
    raw_obj = resp.content
    raw_text = raw_obj if isinstance(raw_obj, str) else str(raw_obj)

    try:
        parsed = raw_obj if isinstance(raw_obj, dict) else _extract_json(raw_text)
    except Exception:
        parsed = {}

    letter = str(parsed.get("answer", "")).strip().upper()
    if letter not in {"A","B","C","D"}:
        alt = str(parsed.get("choice", parsed.get("letter", ""))).strip().upper()
        if alt in {"A","B","C","D"}:
            letter = alt

    if letter not in {"A","B","C","D"}:
        fallback_letter = extract_letter_a_to_d(raw_text) or ""
        if not fallback_letter:
            pairs: list[tuple[str, str]] = []
            for opt in options_list:
                opt = opt.strip()
                if len(opt) >= 3 and opt[1] in ")." and opt[0].upper() in {"A","B","C","D"}:
                    label = opt[0].upper()
                    body = opt[2:].strip()
                    pairs.append((label, body))
                elif ")" in opt:
                    idx = opt.find(")")
                    if idx > 0:
                        label = opt[:idx].strip().upper()
                        body = opt[idx+1:].strip()
                        if label in {"A","B","C","D"}:
                            pairs.append((label, body))
            raw_low = raw_text.lower()
            hits = [lab for (lab, body) in pairs if body and body.lower() in raw_low]
            if len(hits) == 1:
                fallback_letter = hits[0]
        if fallback_letter in {"A","B","C","D"}:
            letter = fallback_letter

    if letter not in {"A","B","C","D"}:
        enforce_llm = _llm(temperature=0.0, json_mode=False, role="answerer")
        enforce_sys = SystemMessage(content=(
            "Return ONLY a single capital letter among A, B, C, D for the question below. No punctuation or explanation."
        ))
        enforce_hum = HumanMessage(content=(
            "Question: " + qstem + "\n"
            + "Options:\n" + options_text + "\n"
        ))
        enforce_resp = enforce_llm.invoke([enforce_sys, enforce_hum])
        enforce_raw = enforce_resp.content if isinstance(enforce_resp.content, str) else str(enforce_resp.content)
        m = __import__("re").search(r"\b([ABCD])\b", enforce_raw.upper())
        letter = m.group(1) if m else ""

    if letter not in {"A","B","C","D"}:
        raise ValueError("Could not extract a valid answer letter A–D from model output.")

    explanation_text = parsed.get("explanation") if isinstance(parsed, dict) else None
    one_sentence = str(explanation_text).strip() if isinstance(explanation_text, str) and explanation_text.strip() else extract_one_sentence(raw_text)
    return {"letter": letter, "one_sentence": one_sentence, "raw": raw_text}


def single_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    explanation = state.get("explanation")
    gpqa_question = state.get("gpqa_question", {})

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Explanation is required in state for single_answer_node.")
    if not isinstance(gpqa_question, Dict) or not gpqa_question:
        raise ValueError("gpqa_question must be a non-empty dictionary in state.")

    res = single_answer(explanation, gpqa_question)
    return {
        "single_answer": res.get("letter", ""),
        "single_explanation": res.get("one_sentence", ""),
    }
