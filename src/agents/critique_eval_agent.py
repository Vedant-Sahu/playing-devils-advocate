from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
import difflib

from src.config.agent_config import _llm, PERSONAS, PERSONA_GUIDELINES
from src.utils.parsing import _extract_json


# Persona importance weights (higher = more critical role)
# These differentiate the value of finding issues in each domain
PERSONA_WEIGHTS: Dict[str, float] = {
    "correctness_validator": 2.0,      # Most critical - wrong method is catastrophic
    "step_sequencer": 1.5,             # Very important - missing steps blocks students
    "assumption_spotter": 1.3,         # Important - hidden assumptions confuse
    "clarity_critic": 1.0,             # Important - unclear = unusable
    "notation_critic": 0.8,            # Helpful but less critical than correctness
}


def _calculate_uniqueness_pairwise(issue: str, other_issues: List[str]) -> float:
    """
    Improved uniqueness: measure if this issue addresses a DIFFERENT ASPECT than others.
    Uses pairwise comparison to find the MOST similar other issue, then returns 1-similarity.
    This rewards finding issues in different domains, not just different phrasing.
    """
    if not issue or not other_issues:
        return 1.0  # Fully unique if no other issues to compare
    
    try:
        # Find the MOST similar other issue (worst case for uniqueness)
        llm = _llm(temperature=0.0, json_mode=True, role="uniqueness_judge")
        
        similarities = []
        for other in other_issues:
            sys = SystemMessage(
                content=(
                    "You are judging if two critiques address the SAME UNDERLYING ISSUE "
                    "even if phrased differently.\n\n"
                    "Rate similarity 0.0-1.0:\n"
                    "- 1.0 = Same issue (e.g., both say 'step 3 is missing')\n"
                    "- 0.5 = Related issues (e.g., both about clarity but different sentences)\n"
                    "- 0.0 = Completely different issues (e.g., one about math error, one about notation)\n\n"
                    "Focus on the SUBSTANCE of the issue, not the phrasing.\n"
                    "Return ONLY JSON: {\"similarity\": 0.0-1.0, \"reasoning\": \"brief\"}"
                )
            )
            
            hum = HumanMessage(
                content=(
                    f"CRITIQUE 1:\n{issue}\n\n"
                    f"CRITIQUE 2:\n{other}\n\n"
                    "How similar are these two critiques in substance?"
                )
            )
            
            resp = llm.invoke([sys, hum])
            raw = resp.content
            parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
            
            similarity = float(parsed.get("similarity", 0.5))
            similarities.append(max(0.0, min(1.0, similarity)))
        
        # Uniqueness is 1 minus the HIGHEST similarity (most similar other issue)
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity
        
    except Exception as e:
        # Fallback: simple string comparison
        similarities = [
            difflib.SequenceMatcher(None, issue.lower(), other.lower()).ratio()
            for other in other_issues
        ]
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity

    
def _judge_severity_with_context(
    issue: str,
    explanation: str,
    persona: str,
    quote: Optional[str] = None,
    previous_iterations_critiques: Optional[List[str]] = None
) -> tuple[int, str, float]:
    """
    Context-aware severity judge that considers persona role and whether issue is new.
    Returns (severity_score, justification, role_alignment).
    
    Severity scale:
    - 3 (critical): Creates major misconceptions, fundamental errors, or makes explanation unusable
    - 2 (moderate): Significant gap that hurts understanding
    - 1 (minor): Small improvement that would help but not essential
    - 0 (no issue): Not actually a problem
    
    Role alignment (0.0-1.0): How well does this critique fit the persona's designated role?
    """
    llm = _llm(temperature=0.0, json_mode=True, role="severity_judge")
    
    persona_context = PERSONA_GUIDELINES.get(persona, "")
    prev_context = ""
    if previous_iterations_critiques:
        prev_context = (
            "\n\nPREVIOUS CRITIQUES FROM EARLIER ITERATIONS:\n" +
            "\n".join([f"- {c}" for c in previous_iterations_critiques[-3:]])  # Last 3 iterations
        )
    
    sys = SystemMessage(
        content=(
            "You are an independent judge evaluating student critiques on physics solution guides.\n\n"
            "Rate on TWO dimensions:\n\n"
            "1. SEVERITY (0-3):\n"
            "   - 3 (critical): Wrong method, major error, or makes guide unusable\n"
            "   - 2 (moderate): Significant gap, missing step, or confusing explanation\n"
            "   - 1 (minor): Small improvement, stylistic issue, or nice-to-have\n"
            "   - 0 (not an issue): Invalid critique or nitpicking\n\n"
            "2. ROLE ALIGNMENT (0.0-1.0):\n"
            "   - 1.0 = Perfectly aligned with this persona's designated role\n"
            "   - 0.5 = Somewhat related to role but not core responsibility\n"
            "   - 0.0 = Completely outside this persona's role\n\n"
            "CRITICAL: Consider if this issue was already raised in previous iterations. "
            "If it's a repeated complaint that wasn't addressed, maintain severity. "
            "If it's a new phrasing of an already-fixed issue, reduce severity.\n\n"
            "Return ONLY JSON: {\"severity\": 0|1|2|3, \"role_alignment\": 0.0-1.0, \"justification\": \"brief reason\"}"
        )
    )
    
    hum = HumanMessage(
        content=(
            f"PERSONA ROLE:\n{persona}\n"
            f"ROLE DESCRIPTION:\n{persona_context}\n\n"
            f"EXPLANATION BEING CRITIQUED:\n{explanation}\n\n"
            f"STUDENT'S CRITIQUE:\n{issue}\n\n"
            f"QUOTED TEXT: {quote or 'N/A'}\n"
            f"{prev_context}\n\n"
            "Evaluate the severity and role alignment of this critique."
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
    
    role_alignment = float(parsed.get("role_alignment", 0.5))
    role_alignment = max(0.0, min(1.0, role_alignment))  # Clamp to [0, 1]
    
    justification = str(parsed.get("justification", "")).strip()
    
    return severity, justification, role_alignment


def _score_and_rank_critiques(
    responses: Dict[str, Any],
    question: str,
    explanation: str,
    student_score_history: Dict[str, List[Dict]]
) -> List[Dict[str, Any]]:
    """
    Improved scoring system with persona-specific weights and role alignment.
    
    Final Score = (Severity × RoleWeight × RoleAlignment) + (Uniqueness × 2.0)
    
    This prevents convergence by:
    - Rewarding personas for staying in their lane (role alignment)
    - Weighting different types of issues differently (persona weights)
    - Using improved pairwise uniqueness calculation
    - Passing ALL critiques to teacher (not just top-K)
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
        
        # Get persona weight (default to 1.0 if not in dict)
        persona_weight = PERSONA_WEIGHTS.get(persona, 1.0)
        
        # Get previous critiques from this persona for context
        prev_critiques = []
        if persona in student_score_history:
            prev_critiques = [
                entry.get("issue", "") 
                for entry in student_score_history[persona][-3:]  # Last 3 iterations
            ]
        
        # If no issue, score is 0
        if issue is None:
            score = 0.0
            validated_severity = 0
            justification = "No issue provided"
            role_alignment = 0.0
            uniqueness = 0.0
            weighted_severity = 0.0
        else:
            # Judge independently assigns severity and role alignment
            validated_severity, justification, role_alignment = _judge_severity_with_context(
                issue, explanation, persona, quote, prev_critiques
            )
            
            # Weighted severity component
            # severity (0-3) × persona_weight (0.8-2.0) × role_alignment (0-1)
            weighted_severity = float(validated_severity) * persona_weight * role_alignment
            
            # Calculate uniqueness compared to other students' issues (pairwise)
            other_issues = [
                responses[p].get("issue")
                for p in responses
                if p != persona and responses[p].get("issue") is not None
            ]
            
            uniqueness = _calculate_uniqueness_pairwise(issue, other_issues)
            
            # Uniqueness bonus (up to 2.0 points - reduced from 3.0 to balance with severity)
            uniqueness_bonus = uniqueness * 2.0
            
            # Final score
            score = weighted_severity + uniqueness_bonus
        
        scored.append({
            "persona": persona,
            "feedback": feedback,
            "score": score,
            "validated_severity": validated_severity,
            "severity_justification": justification,
            "role_alignment": role_alignment,
            "persona_weight": persona_weight,
            "weighted_severity": weighted_severity,
            "uniqueness": uniqueness,
            "uniqueness_bonus": uniqueness_bonus if issue else 0.0
        })
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    # Assign ranks
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank
    
    return scored


def _format_all_critiques(scored: List[Dict[str, Any]]) -> str:
    """
    Format ALL critiques for the teacher agent (not just top-K).
    Ranked by score but all personas get their feedback through.
    
    This prevents personas from being silenced, which was causing convergent behavior.
    """
    # Filter to only those with actual issues (severity > 0)
    valid_issues = [
        s for s in scored 
        if s["validated_severity"] > 0
    ]
    
    if not valid_issues:
        return "No significant issues identified by students."
    
    result = "STUDENT FEEDBACK (ranked by score):\n\n"
    for i, item in enumerate(valid_issues, 1):
        fb = item["feedback"]
        persona_display = item['persona'].replace('_', ' ').title()
        
        result += f"{i}. [{persona_display}] (rank #{item['rank']}, score {item['score']:.2f}):\n"
        result += f"   Issue: {fb['issue']}\n"
        if fb.get('quote'):
            result += f"   Quote: \"{fb['quote']}\"\n"
        
        # Show scoring breakdown for transparency
        result += (
            f"   Severity: {item['validated_severity']}/3 "
            f"(weight={item['persona_weight']:.1f}, "
            f"role_align={item['role_alignment']:.2f}) = {item['weighted_severity']:.2f} pts\n"
        )
        result += f"   Uniqueness: {item['uniqueness']:.2f} = +{item['uniqueness_bonus']:.2f} pts\n"
        result += f"   Judge's reasoning: {item['severity_justification']}\n\n"
    
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
            "role_alignment": item.get("role_alignment", 0.0),
            "persona_weight": item.get("persona_weight", 1.0),
            "weighted_severity": item.get("weighted_severity", 0.0),
            "uniqueness": item.get("uniqueness", 0.0),
            "uniqueness_bonus": item.get("uniqueness_bonus", 0.0)
        })
    
    return current_history


def reward_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improved scoring system that prevents convergence through:
    1. Persona-specific weights (correctness > clarity > notation)
    2. Role alignment scores (rewards staying in lane)
    3. Pairwise uniqueness (rewards finding different issues)
    4. Passing ALL feedback to teacher (not just top-K)
    
    Score formula:
    Final = (Severity × PersonaWeight × RoleAlignment) + (Uniqueness × 2.0)
    
    This creates natural specialization and prevents all personas from
    chasing the same high-severity issues.
    """
    responses = state.get("student_responses", {})
    if not isinstance(responses, dict):
        raise ValueError("student_responses must be a dict.")
    
    question = str(state.get("question", ""))
    explanation = str(state.get("explanation", ""))
    student_score_history = state.get("student_score_history", {})
    
    # Score and rank all critiques with improved system
    scored = _score_and_rank_critiques(responses, question, explanation, student_score_history)
    
    # Format ALL critiques for teacher (not just top-K)
    filtered_critiques = _format_all_critiques(scored)
    
    # Update score history for all students
    updated_history = _update_score_history(state, scored)
    
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
        "filtered_critiques": filtered_critiques  # What teacher sees (now includes all)
    })
    
    return {
        "reward_scores": reward_scores,
        "filtered_critiques": filtered_critiques,
        "critique_rankings": scored,
        "student_score_history": updated_history,
        "history": history
    }