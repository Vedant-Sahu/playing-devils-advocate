from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.schema import HumanMessage, SystemMessage
from src.config.agent_config import _llm
from src.utils.parsing import _extract_json


def pairwise_judge_educational(
    question: str,
    expert_explanation: str,
    explanation_a: str,
    explanation_b: str,
    label_a: str = "adaptive",
    label_b: str = "baseline"
) -> Dict[str, Any]:
    """
    Compare two educational explanations using physics education criteria.
    
    Args:
        question: The physics question being explained
        expert_explanation: Reference explanation with correct physics content
        explanation_a: First explanation to compare
        explanation_b: Second explanation to compare
        label_a: Label for explanation A (e.g., "adaptive")
        label_b: Label for explanation B (e.g., "baseline")
    
    Returns:
        Dict containing winner, rationales, and educational quality scores
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(explanation_a, str) or not explanation_a.strip():
        raise ValueError("explanation_a must be a non-empty string.")
    if not isinstance(explanation_b, str) or not explanation_b.strip():
        raise ValueError("explanation_b must be a non-empty string.")
    if not isinstance(expert_explanation, str) or not expert_explanation.strip():
        raise ValueError("expert_explanation must be a non-empty string.")

    llm = _llm(temperature=0.3, json_mode=True, role="pairwise_judge")  # Lower temp for consistency

    # Physics education-specific evaluation criteria
    sys = SystemMessage(
        content="""You are an expert Physics Education Researcher evaluating explanations for graduate-level physics questions.

            Your task: Compare two explanations head-to-head using THREE equally-important criteria. The explanation that wins on 2 or more criteria is the winner.

            EVALUATION CRITERIA (all equally weighted):

            1. **Physics Correctness**
            - Contains no physics misconceptions or errors
            - Correctly represents fundamental principles
            - Uses precise physics terminology
            - Aligns with the expert explanation on key concepts (use expert explanation to verify correctness, NOT as a completeness standard)
            
            Winner: Which explanation has fewer physics errors?

            2. **Pedagogical Quality**
            - Builds understanding progressively (simple → complex)
            - Explains WHY, not just WHAT
            - Clear logical flow between concepts
            - Appropriate level for graduate physics students
            - Uses effective analogies or examples (if present)
            
            Winner: Which explanation would better help a student learn?

            3. **Clarity & Precision**
            - Well-structured and organized
            - Appropriately detailed (not too verbose, not too sparse)
            - Clear mathematical reasoning (if applicable)
            - Unambiguous language
            - Addresses potential sources of confusion
            
            Winner: Which explanation is clearer and more precise?

            DECISION RULE:
            - Compare explanations on each criterion independently
            - The explanation that wins on 2 or more criteria is the overall winner
            - Only declare "tie" if each explanation wins 1 criterion and 1 criterion is truly tied, OR if all 3 criteria are tied

            OUTPUT FORMAT (JSON only):
            {
                "criterion_winners": {
                    "physics_correctness": "A" | "B" | "tie",
                    "pedagogical_quality": "A" | "B" | "tie",
                    "clarity_precision": "A" | "B" | "tie"
                },
                "winner": "A" | "B" | "tie",
                "rationales": {
                    "physics_correctness": "Concise comparison explaining why one is more correct. Cite specific examples.",
                    "pedagogical_quality": "Concise comparison explaining which teaches better. Cite specific examples.",
                    "clarity_precision": "Concise comparison explaining which is clearer. Cite specific examples."
                }
            }

            IMPORTANT:
            - Be decisive - avoid "tie" unless truly indistinguishable
            - Use the expert explanation only to check correctness, not to judge completeness
            - Cite specific phrases/concepts from the explanations in your rationales"""
    )

    # Construct evaluation payload
    payload = {
        "question": question,
        "expert_explanation": expert_explanation,
        "explanation_A": {
            "label": str(label_a),
            "text": explanation_a
        },
        "explanation_B": {
            "label": str(label_b),
            "text": explanation_b
        }
    }
    
    hum = HumanMessage(content=str(payload))

    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    if not isinstance(parsed, dict):
        raise ValueError("Pairwise Judge must return a JSON object.")

    # Validate response structure
    winner = parsed.get("winner", "").strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"'winner' must be one of 'A', 'B', 'tie', got: {winner}")

    criterion_winners = parsed.get("criterion_winners", {})
    if not isinstance(criterion_winners, dict):
        raise ValueError("Response must include 'criterion_winners' dictionary")

    rationales = parsed.get("rationales", {})
    if not isinstance(rationales, dict):
        raise ValueError("Response must include 'rationales' dictionary")

    # Verify 2-out-of-3 logic
    a_wins = sum(1 for v in criterion_winners.values() if str(v).upper() == "A")
    b_wins = sum(1 for v in criterion_winners.values() if str(v).upper() == "B")
    
    expected_winner = "A" if a_wins >= 2 else "B" if b_wins >= 2 else "tie"
    if winner != expected_winner.upper():
        print(f"Warning: Winner '{winner}' doesn't match 2-out-of-3 logic. Expected '{expected_winner}'. Using criterion count.")
        winner = expected_winner.upper()

    return {
        "winner": winner if winner != "TIE" else "tie",
        "criterion_winners": {k: str(v).upper() if str(v).upper() != "TIE" else "tie" for k, v in criterion_winners.items()},
        "rationales": rationales,
        "criterion_scores": {
            "A": a_wins,
            "B": b_wins
        }
    }


def batch_pairwise_comparison(
    questions: List[str],
    expert_explanations: List[str],
    explanations_a: List[str],
    explanations_b: List[str],
    label_a: str = "adaptive",
    label_b: str = "baseline",
) -> Dict[str, Any]:
    """
    Run pairwise comparisons on multiple question-explanation pairs.
    
    Returns aggregate statistics across all comparisons.
    """
    if not (len(questions) == len(expert_explanations) == len(explanations_a) == len(explanations_b)):
        raise ValueError("All input lists must have the same length")
    
    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    
    # Track criterion-level wins
    criterion_wins_a = {"physics_correctness": 0, "pedagogical_quality": 0, "clarity_precision": 0}
    criterion_wins_b = {"physics_correctness": 0, "pedagogical_quality": 0, "clarity_precision": 0}
    
    for i, (q, exp, a, b) in enumerate(zip(questions, expert_explanations, explanations_a, explanations_b)):
        try:
            result = pairwise_judge_educational(q, exp, a, b, label_a, label_b)
            results.append(result)
            
            # Overall winner count
            if result["winner"] == "A":
                wins_a += 1
            elif result["winner"] == "B":
                wins_b += 1
            else:
                ties += 1
            
            # Criterion-level tracking
            for criterion, winner in result["criterion_winners"].items():
                if winner == "A":
                    criterion_wins_a[criterion] += 1
                elif winner == "B":
                    criterion_wins_b[criterion] += 1
            
        except Exception as e:
            print(f"Error processing comparison {i}: {e}")
            results.append({"error": str(e)})
    
    total = len(questions)
    return {
        "individual_results": results,
        "summary": {
            "total_comparisons": total,
            f"{label_a}_wins": wins_a,
            f"{label_b}_wins": wins_b,
            "ties": ties,
            f"{label_a}_win_rate": wins_a / total if total > 0 else 0,
            f"{label_b}_win_rate": wins_b / total if total > 0 else 0,
            "criterion_breakdown": {
                f"{label_a}": criterion_wins_a,
                f"{label_b}": criterion_wins_b,
            }
        }
    }