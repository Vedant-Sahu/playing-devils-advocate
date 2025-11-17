from __future__ import annotations
from typing import Any, Dict
from langchain.schema import HumanMessage, SystemMessage
import difflib

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import _extract_json


def _calculate_uniqueness(issue: str, other_issues: List[str]) -> float:
    """
    Measure how unique this critique is compared to others using LLM-based similarity.
    Returns a score between 0.0 (identical to others) and 1.0 (completely unique).
    """
    if not issue or not other_issues:
        return 1.0  # Fully unique if no other issues to compare
    
    try:
        # Use LLM to judge similarity instead of embeddings
        llm = _llm(temperature=0.0, json_mode=True, role="uniqueness_judge")
        
        # Format other issues
        other_issues_text = "\n".join([
            f"- {other}" for other in other_issues
        ])
        
        sys = SystemMessage(
            content=(
                "You are judging how unique a critique is compared to others.\n"
                "Rate uniqueness 0.0-1.0:\n"
                "- 1.0 = Completely unique, different angle/issue\n"
                "- 0.5 = Somewhat similar, overlapping concerns\n"
                "- 0.0 = Nearly identical to another critique\n\n"
                "Return ONLY JSON: {\"uniqueness\": 0.0-1.0, \"reasoning\": \"brief\"}"
            )
        )
        
        hum = HumanMessage(
            content=(
                f"CRITIQUE TO EVALUATE:\n{issue}\n\n"
                f"OTHER CRITIQUES:\n{other_issues_text}\n\n"
                "How unique is the first critique compared to the others?"
            )
        )
        
        resp = llm.invoke([sys, hum])
        raw = resp.content
        parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
        
        uniqueness = float(parsed.get("uniqueness", 0.5))
        return max(0.0, min(1.0, uniqueness))  # Clamp to [0, 1]
        
    except Exception as e:
        # Fallback: simple string comparison
        similarities = [
            difflib.SequenceMatcher(None, issue.lower(), other.lower()).ratio()
            for other in other_issues
        ]
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity

    
def _judge_severity(
    issue: str,
    explanation: str,
    quote: Optional[str] = None
) -> tuple[int, str]:
    """
    Independent judge evaluates the severity of a student's critique.
    Returns (severity_score, justification).
    
    Severity scale:
    - 3 (critical): Creates major misconceptions or makes explanation unusable
    - 2 (moderate): Significant gap that hurts understanding
    - 1 (minor): Small improvement that would help but not essential
    - 0 (no issue): Not actually a problem
    """
    llm = _llm(temperature=0.0, json_mode=True, role="severity_judge")
    
    sys = SystemMessage(
        content=(
            "You are an independent judge evaluating the severity of student critiques "
            "on physics explanations.\n\n"
            "Rate severity 1-3:\n"
            "- 3 (critical): Creates major misconceptions, fundamental errors, or makes explanation unusable\n"
            "- 2 (moderate): Significant gap or confusion that hurts understanding\n"
            "- 1 (minor): Small improvement that would help but not essential\n"
            "- 0 (not an issue): The critique is invalid or trivial\n\n"
            "Be skeptical and fair. Most issues are severity 1-2, not 3.\n"
            "Return ONLY JSON: {\"severity\": 0|1|2|3, \"justification\": \"brief reason\"}"
        )
    )
    
    hum = HumanMessage(
        content=(
            f"EXPLANATION:\n{explanation}\n\n"
            f"STUDENT'S CRITIQUE:\n{issue}\n\n"
            f"QUOTED TEXT: {quote or 'N/A'}\n\n"
            "Evaluate the severity of this critique."
        )
    )
    
    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    if not isinstance(parsed, dict):
        raise ValueError("Severity judge must return JSON object.")
    
    severity = int(parsed.get("severity", 1))
    if severity not in {0, 1, 2, 3}:
        severity = 1  # Default to minor if invalid
    
    justification = str(parsed.get("justification", "")).strip()
    
    return severity, justification


def _score_and_rank_critiques(
    responses: Dict[str, Any],
    question: str,
    explanation: str
) -> List[Dict[str, Any]]:
    """
    Score all student critiques using judge-assigned severity and uniqueness.
    Returns list of scored items sorted by final score (highest first).
    """
    scored = []
    
    # Collect all non-null issues for uniqueness calculation
    all_issues = [
        fb.get("issue") 
        for fb in responses.values() 
        if fb.get("issue") is not None
    ]
    
    for persona, feedback in responses.items():
        issue = feedback.get("issue")
        quote = feedback.get("quote")
        
        # If no issue, score is 0
        if issue is None:
            score = 0.0
            validated_severity = 0
            justification = "No issue provided"
            uniqueness = 0.0
            uniqueness_bonus = 0.0
        else:
            # Judge independently assigns severity
            validated_severity, justification = _judge_severity(
                issue, explanation, quote
            )
            
            # Base score from validated severity (0-3)
            base_score = float(validated_severity)
            
            # Calculate uniqueness compared to other students' issues
            other_issues = [
                responses[p].get("issue")
                for p in responses
                if p != persona and responses[p].get("issue") is not None
            ]
            
            uniqueness = _calculate_uniqueness(issue, other_issues)
            
            # Uniqueness bonus (up to 3 points)
            uniqueness_bonus = uniqueness * 3.0
            
            # Final score
            score = base_score + uniqueness_bonus
        
        scored.append({
            "persona": persona,
            "feedback": feedback,
            "score": score,
            "validated_severity": validated_severity,
            "severity_justification": justification,
            "uniqueness": uniqueness,
            "uniqueness_bonus": uniqueness_bonus
        })
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    # Assign ranks
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank
    
    return scored


def _format_filtered_critiques(scored: List[Dict[str, Any]], top_k: int = 3) -> str:
    """
    Format the top-K critiques for the teacher agent.
    """
    # Filter to only those with actual issues (severity > 0)
    top_issues = [
        s for s in scored 
        if s["validated_severity"] > 0
    ][:top_k]
    
    if not top_issues:
        return "No significant issues identified by students."
    
    result = "TOP STUDENT CONCERNS:\n\n"
    for i, item in enumerate(top_issues, 1):
        fb = item["feedback"]
        result += f"{i}. [{item['persona']}] (severity {item['validated_severity']}/3, score {item['score']:.1f}):\n"
        result += f"   Issue: {fb['issue']}\n"
        if fb.get('quote'):
            result += f"   Quote: \"{fb['quote']}\"\n"
        result += f"   Judge's reasoning: {item['severity_justification']}\n"
        result += f"   Rank: #{item['rank']}/5 students\n\n"
    
    return result


def _update_score_history(
    state: Dict[str, Any],
    scored: List[Dict[str, Any]]
) -> Dict[str, List[Dict]]:
    """
    Update the score history for each student persona.
    """
    current_history = state.get("student_score_history", {})
    current_iteration = state.get("iteration", 0)
    
    for item in scored:
        persona = item["persona"]
        if persona not in current_history:
            current_history[persona] = []
        
        current_history[persona].append({
            "iteration": current_iteration,
            "score": item["score"],
            "rank": item["rank"],
            "issue": item["feedback"].get("issue") or "No issue found",
            "validated_severity": item["validated_severity"],
            "severity_justification": item["severity_justification"],
            "uniqueness": item.get("uniqueness", 0.0),
            "uniqueness_bonus": item.get("uniqueness_bonus", 0.0)
        })
    
    return current_history


def reward_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score and rank student critiques with judge-assigned severity, 
    select top-K for teacher, and update history.
    
    Competitive scoring:
    - Base score: Judge-validated severity (0-3 points) - prevents gaming
    - Uniqueness bonus: Up to 3 points for finding issues others don't
    - Only top-3 critiques are passed to teacher
    """
    responses = state.get("student_responses", {})
    if not isinstance(responses, dict):
        raise ValueError("student_responses must be a dict.")
    
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    
    # Score and rank all critiques (includes judge calls for severity)
    scored = _score_and_rank_critiques(responses, question, explanation)
    
    # Format top-K critiques for teacher
    filtered_critiques = _format_filtered_critiques(scored, top_k=3)
    
    # Update score history for all students
    student_score_history = _update_score_history(state, scored)
    
    # Extract scores dict for easy access
    reward_scores = {item["persona"]: item["score"] for item in scored}
    
    # Update history
    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "explanation": explanation,
        "student_responses": responses,
        "reward_scores": reward_scores,
        "critique_rankings": scored,  # Full ranking info with judge validations
        "filtered_critiques": filtered_critiques  # What teacher sees
    })
    
    return {
        "reward_scores": reward_scores,
        "filtered_critiques": filtered_critiques,
        "critique_rankings": scored,
        "student_score_history": student_score_history,
        "history": history
    }