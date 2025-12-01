"""
Answer Leakage Checker Agent - Detects if teacher explanation leaks the answer.

This agent reviews the teacher's explanation to ensure it doesn't directly reveal
the correct answer, specific answer values, or use numbers from the question that
would make the answer obvious.
"""

from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.agent_config import _llm
from src.utils.parsing import _extract_json


def check_answer_leakage(
    explanation: str,
    question: str,
    correct_answer: str,
    options: list[str]
) -> Dict[str, Any]:
    """
    Check if the explanation leaks the correct answer.
    
    Args:
        explanation: Teacher's explanation to check
        question: The original question
        correct_answer: The correct answer (letter or value)
        options: List of answer options
        
    Returns:
        Dictionary with:
        - leakage_detected: bool (True if leakage found)
        - feedback: str (what needs to be fixed)
        - confidence: float (0-1, how confident about the detection)
    """
    llm = _llm(temperature=0.0, json_mode=True, role="answer_leak_checker", max_tokens=1000)
    
    sys = SystemMessage(
        content=(
            "You are an answer leakage detector. Your job is to determine if a teacher's "
            "explanation reveals the correct answer to a question.\n\n"
            
            "ANSWER LEAKAGE includes:\n"
            "1. Directly stating the correct answer letter (e.g., 'The answer is B')\n"
            "2. Stating the exact numerical value of the correct answer\n"
            "3. Using specific numbers from the question in worked examples that lead to the answer\n"
            "4. Providing step-by-step calculations using the question's actual values\n"
            "5. Describing the correct option in a way that makes it obvious (e.g., 'The correct choice is approximately 8.33 years')\n\n"
            
            "NOT LEAKAGE:\n"
            "1. Teaching the general method/approach to solve this type of problem\n"
            "2. Explaining relevant concepts and formulas\n"
            "3. Using DIFFERENT numbers in generic examples\n"
            "4. Describing what type of calculation or reasoning to apply\n"
            "5. Explaining the physics/theory without working through the specific problem\n\n"
            
            "Return JSON with:\n"
            "{\n"
            '  "leakage_detected": true/false,\n'
            '  "feedback": "Specific description of what leaked (or empty string if no leakage)",\n'
            '  "confidence": 0.0-1.0 (how confident you are)\n'
            "}\n\n"
            
            "Be strict but fair. Generic teaching is fine. Revealing the answer is not."
        )
    )
    
    hum = HumanMessage(
        content=(
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n" + "\n".join(options) + f"\n\n"
            f"CORRECT ANSWER: {correct_answer}\n\n"
            f"TEACHER EXPLANATION:\n{explanation}\n\n"
            "Does this explanation leak the answer? Return JSON only."
        )
    )
    
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    if not isinstance(parsed, dict):
        # Fallback: assume no leakage if parsing fails
        return {
            "leakage_detected": False,
            "feedback": "",
            "confidence": 0.0
        }
    
    return {
        "leakage_detected": parsed.get("leakage_detected", False),
        "feedback": parsed.get("feedback", ""),
        "confidence": parsed.get("confidence", 0.0)
    }


def check_answer_leakage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Graph node that checks for answer leakage and updates state.
    
    If leakage is detected, sets answer_leakage_detected=True and provides
    feedback to teacher. Also updates the single_student_critique or
    filtered_critiques to include the leakage feedback.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with leakage detection results
    """
    explanation = state.get("explanation", "")
    gpqa_question = state.get("gpqa_question", {})
    
    question = gpqa_question.get("question", "")
    correct_answer = gpqa_question.get("correct", "") or gpqa_question.get("correct_answer", "")
    options = gpqa_question.get("options", [])
    
    if not explanation or not question:
        # No explanation or question to check
        return {
            "answer_leakage_detected": False,
            "leakage_feedback": ""
        }
    
    # Run leakage detection
    result = check_answer_leakage(explanation, question, correct_answer, options)
    
    leakage_detected = result["leakage_detected"]
    feedback = result["feedback"]
    
    if leakage_detected:
        print(f"  ⚠️  ANSWER LEAKAGE DETECTED (confidence: {result['confidence']:.2f})")
        print(f"      {feedback}")
        
        # Build feedback message for teacher
        leakage_feedback = (
            f"CRITICAL ISSUE - Answer Leakage Detected:\n"
            f"{feedback}\n\n"
            f"You must revise your explanation to remove any information that reveals "
            f"the correct answer. Teach the method generically without using the specific "
            f"numbers from this question or revealing the final answer value."
        )
        
        # Update the appropriate critique field based on mode
        # For single student mode
        if "single_student_critique" in state:
            return {
                "answer_leakage_detected": True,
                "leakage_feedback": leakage_feedback,
                "single_student_critique": leakage_feedback,
                "decision": "CONTINUE"  # Override STOP decision
            }
        # For multi-student mode
        else:
            return {
                "answer_leakage_detected": True,
                "leakage_feedback": leakage_feedback,
                "filtered_critiques": leakage_feedback,
                "decision": "CONTINUE"  # Override STOP decision
            }
    else:
        print(f"  ✓ No answer leakage detected (confidence: {result['confidence']:.2f})")
        return {
            "answer_leakage_detected": False,
            "leakage_feedback": ""
        }