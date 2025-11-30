from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.agent_config import _llm
from src.utils.parsing import _extract_json


def multi_metric_judge_educational(
    question: str,
    correct_answer: str,
    explanation_a: str,
    explanation_b: str,
    label_a: str = "adaptive",
    label_b: str = "baseline",
    expert_explanation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two educational explanations using comprehensive physics education metrics.
    
    Args:
        question: The physics question being explained
        correct_answer: The correct answer to the question
        explanation_a: First explanation to compare
        explanation_b: Second explanation to compare
        label_a: Label for explanation A (e.g., "adaptive")
        label_b: Label for explanation B (e.g., "baseline")
        expert_explanation: Optional reference explanation for additional context
    
    Returns:
        Dict containing per-metric scores (0-10), reasoning, and overall comparison
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        raise ValueError("correct_answer must be a non-empty string.")
    if not isinstance(explanation_a, str) or not explanation_a.strip():
        raise ValueError("explanation_a must be a non-empty string.")
    if not isinstance(explanation_b, str) or not explanation_b.strip():
        raise ValueError("explanation_b must be a non-empty string.")

    llm = _llm(temperature=0.2, json_mode=True, role="multi_metric_judge")

    # Comprehensive multi-metric evaluation system
    sys = SystemMessage(
        content="""You are an expert Physics Education Researcher evaluating explanations for graduate-level physics questions.

Your task: Evaluate two explanations across 6 independent metrics. For each metric:
1. First, provide detailed chain-of-thought reasoning
2. Then assign a score from 0-10 (integers only)
3. Score BOTH explanations independently on each metric

CRITICAL: You have access to the CORRECT ANSWER. Your primary job is to evaluate whether each explanation guides students toward this correct answer or introduces misconceptions/red herrings that could mislead them.

════════════════════════════════════════════════════════════════════════════════
EVALUATION METRICS (score each explanation independently 0-10):
════════════════════════════════════════════════════════════════════════════════

1. **CONCEPTUAL ACCURACY** (Does it lead to the correct answer?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Completely accurate, directly supports correct answer, no errors
   • 7-8:  Accurate overall, only minor imprecisions that don't affect correctness
   • 5-6:  Mostly correct but contains some inaccuracies or unclear points
   • 3-4:  Mix of correct and incorrect information, could confuse students
   • 1-2:  Significant errors that mislead students away from correct answer
   • 0:     Fundamentally incorrect or contradicts the correct answer
   
   Evaluate:
   - Does the explanation lead toward the CORRECT ANSWER provided?
   - Are all physics principles correctly stated?
   - Are there any factual errors or misconceptions?
   - Does it use precise physics terminology?
   - Would following this explanation lead to the correct answer?

2. **PEDAGOGICAL CLARITY** (How well does it teach?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Exceptional teaching - clear progression, effective scaffolding, explains WHY
   • 7-8:  Strong pedagogy with logical flow and good explanations
   • 5-6:  Adequate teaching but could be more systematic or explanatory
   • 3-4:  Some pedagogical elements but confusing or incomplete flow
   • 1-2:  Poor structure, jumps between concepts without connection
   • 0:     Chaotic or incomprehensible pedagogically
   
   Evaluate:
   - Does it build understanding progressively (simple → complex)?
   - Does it explain WHY, not just WHAT?
   - Is there clear logical flow between concepts?
   - Are examples/analogies helpful (if present)?
   - Would a student understand the reasoning process?

3. **MISCONCEPTION AVOIDANCE** (Does it prevent student confusion?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Anticipates and addresses common misconceptions, no red herrings
   • 7-8:  Avoids misconceptions effectively, clear throughout
   • 5-6:  Generally safe but could be clearer on potentially confusing points
   • 3-4:  Contains some misleading elements or ambiguities
   • 1-2:  Introduces misconceptions or significant red herrings
   • 0:     Multiple serious misconceptions that would confuse students
   
   Evaluate:
   - Does it introduce any RED HERRINGS that distract from the correct answer?
   - Does it address or avoid common physics misconceptions?
   - Could any part mislead students away from the correct answer?
   - Are potentially confusing points clarified?
   - Does it prevent common student errors?

4. **COMPLETENESS** (Does it cover what's needed?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Comprehensive coverage of all essential concepts at appropriate depth
   • 7-8:  Covers key points well, only minor gaps that don't affect understanding
   • 5-6:  Addresses main concepts but missing some important details
   • 3-4:  Significant gaps in coverage that affect understanding
   • 1-2:  Superficial treatment, many important concepts missing
   • 0:     Severely incomplete or off-topic
   
   Evaluate:
   - Are all concepts needed to reach the CORRECT ANSWER explained?
   - Is the depth appropriate for graduate-level physics?
   - Are important steps or reasoning omitted?
   - Does it provide sufficient context and background?

5. **ACCESSIBILITY** (Is it appropriate for the audience?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Perfectly pitched for graduate physics students - clear yet rigorous
   • 7-8:  Appropriate level with good balance of clarity and sophistication
   • 5-6:  Generally appropriate but some sections too simple or too complex
   • 3-4:  Often inappropriate level (too basic or too advanced)
   • 1-2:  Significantly mismatched to graduate student audience
   • 0:     Completely inappropriate level
   
   Evaluate:
   - Is language appropriate for graduate physics students?
   - Does it assume appropriate background knowledge?
   - Is mathematical notation used correctly and clearly?
   - Are technical terms defined when necessary but not over-explained?

6. **ENGAGEMENT POTENTIAL** (Would it maintain student interest?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Highly engaging - compelling narrative, interesting examples, maintains focus
   • 7-8:  Engaging with good structure and relevant connections
   • 5-6:  Adequate engagement, straightforward but not boring
   • 3-4:  Somewhat dry or hard to follow, could lose student attention
   • 1-2:  Disengaging, tedious, or unnecessarily convoluted
   • 0:     Would actively discourage learning
   
   Evaluate:
   - Does it connect concepts in interesting ways?
   - Are examples relevant and illuminating?
   - Does the structure maintain focus and momentum?
   - Would a student stay motivated while reading this?

════════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (JSON only):
════════════════════════════════════════════════════════════════════════════════

{
  "explanation_A_scores": {
    "conceptual_accuracy": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence chain-of-thought explaining the score. Cite specific examples from the explanation.>"
    },
    "pedagogical_clarity": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation with specific examples>"
    },
    "misconception_avoidance": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation. Note any red herrings or misconceptions.>"
    },
    "completeness": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation with specific gaps or strengths>"
    },
    "accessibility": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation of appropriateness for audience>"
    },
    "engagement_potential": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation of engagement quality>"
    }
  },
  "explanation_B_scores": {
    "conceptual_accuracy": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence chain-of-thought explaining the score. Cite specific examples.>"
    },
    "pedagogical_clarity": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation with specific examples>"
    },
    "misconception_avoidance": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation. Note any red herrings or misconceptions.>"
    },
    "completeness": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation with specific gaps or strengths>"
    },
    "accessibility": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation of appropriateness for audience>"
    },
    "engagement_potential": {
      "score": <integer 0-10>,
      "reasoning": "<2-3 sentence explanation of engagement quality>"
    }
  },
  "comparative_summary": {
    "overall_winner": "A" | "B" | "tie",
    "winner_rationale": "<3-4 sentence summary explaining which explanation is better overall and why. Consider all metrics holistically.>",
    "key_differences": "<2-3 sentences highlighting the most important differences between the explanations.>"
  }
}

════════════════════════════════════════════════════════════════════════════════
EVALUATION INSTRUCTIONS:
════════════════════════════════════════════════════════════════════════════════

1. For EACH metric, evaluate BOTH explanations INDEPENDENTLY (don't compare directly yet)
2. Use chain-of-thought reasoning BEFORE assigning scores
3. Be specific - cite exact phrases/concepts from the explanations
4. Check accuracy against the CORRECT ANSWER - this is paramount
5. Identify any red herrings or misconceptions explicitly
6. Assign integer scores 0-10 based on the rubrics
7. In comparative_summary, determine overall winner by considering:
   - Average scores across metrics
   - Critical importance of Conceptual Accuracy and Misconception Avoidance
   - Whether explanation leads students to the correct answer
8. Only declare "tie" if scores are very close (within 5 total points) and both have similar strengths/weaknesses

IMPORTANT:
- An explanation with perfect pedagogy but wrong physics should score low overall
- An explanation that introduces red herrings away from the correct answer should score very low on Misconception Avoidance
- Be decisive but fair - use the full 0-10 range when appropriate"""
    )

    # Construct evaluation payload
    payload = {
        "question": question,
        "correct_answer": correct_answer,
        "explanation_A": {
            "label": str(label_a),
            "text": explanation_a
        },
        "explanation_B": {
            "label": str(label_b),
            "text": explanation_b
        }
    }
    
    if expert_explanation:
        payload["expert_explanation_reference"] = expert_explanation

    hum = HumanMessage(content=str(payload))

    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    if not isinstance(parsed, dict):
        raise ValueError("Multi-metric Judge must return a JSON object.")

    # Validate response structure
    for key in ["explanation_A_scores", "explanation_B_scores", "comparative_summary"]:
        if key not in parsed:
            raise ValueError(f"Response must include '{key}'")

    # Calculate aggregate scores
    metrics = ["conceptual_accuracy", "pedagogical_clarity", "misconception_avoidance", 
               "completeness", "accessibility", "engagement_potential"]
    
    scores_a = parsed["explanation_A_scores"]
    scores_b = parsed["explanation_B_scores"]
    
    # Validate all metrics present and scores are 0-10
    for exp_scores, label in [(scores_a, "A"), (scores_b, "B")]:
        for metric in metrics:
            if metric not in exp_scores:
                raise ValueError(f"Missing metric '{metric}' for explanation {label}")
            score = exp_scores[metric].get("score")
            if not isinstance(score, int) or not (0 <= score <= 10):
                raise ValueError(f"Score for {metric} (explanation {label}) must be integer 0-10, got {score}")

    # Calculate totals and averages
    total_a = sum(scores_a[m]["score"] for m in metrics)
    total_b = sum(scores_b[m]["score"] for m in metrics)
    avg_a = total_a / len(metrics)
    avg_b = total_b / len(metrics)
    
    winner = parsed["comparative_summary"].get("overall_winner", "").strip().upper()
    if winner not in {"A", "B", "TIE"}:
        # Auto-determine based on scores if not provided correctly
        if abs(total_a - total_b) <= 5:  # Within 5 points = tie
            winner = "TIE"
        else:
            winner = "A" if total_a > total_b else "B"
        print(f"Warning: Invalid winner '{winner}', auto-determined as {winner} based on scores")

    return {
        "label_a": label_a,
        "label_b": label_b,
        "scores": {
            label_a: scores_a,
            label_b: scores_b
        },
        "aggregates": {
            label_a: {
                "total_score": total_a,
                "average_score": round(avg_a, 2),
                "max_possible": len(metrics) * 10
            },
            label_b: {
                "total_score": total_b,
                "average_score": round(avg_b, 2),
                "max_possible": len(metrics) * 10
            }
        },
        "comparative_summary": parsed["comparative_summary"],
        "winner": winner.lower() if winner != "TIE" else "tie",
        "score_difference": total_a - total_b
    }


def batch_multi_metric_comparison(
    questions: List[str],
    correct_answers: List[str],
    explanations_a: List[str],
    explanations_b: List[str],
    label_a: str = "adaptive",
    label_b: str = "baseline",
    expert_explanations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run multi-metric comparisons on multiple question-explanation pairs.
    
    Returns aggregate statistics across all comparisons.
    """
    if not (len(questions) == len(correct_answers) == len(explanations_a) == len(explanations_b)):
        raise ValueError("All input lists must have the same length")
    
    if expert_explanations is not None and len(expert_explanations) != len(questions):
        raise ValueError("expert_explanations must match length of questions")

    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    
    # Track metric-level aggregates
    metrics = ["conceptual_accuracy", "pedagogical_clarity", "misconception_avoidance", 
               "completeness", "accessibility", "engagement_potential"]
    
    metric_totals_a = {m: 0 for m in metrics}
    metric_totals_b = {m: 0 for m in metrics}
    metric_wins_a = {m: 0 for m in metrics}
    metric_wins_b = {m: 0 for m in metrics}
    
    total_score_a = 0
    total_score_b = 0
    
    for i, (q, ans, a, b) in enumerate(zip(questions, correct_answers, explanations_a, explanations_b)):
        try:
            expert = expert_explanations[i] if expert_explanations else None
            result = multi_metric_judge_educational(q, ans, a, b, label_a, label_b, expert)
            results.append(result)
            
            # Overall winner count
            if result["winner"] == "a":
                wins_a += 1
            elif result["winner"] == "b":
                wins_b += 1
            else:
                ties += 1
            
            # Aggregate scores
            total_score_a += result["aggregates"][label_a]["total_score"]
            total_score_b += result["aggregates"][label_b]["total_score"]
            
            # Per-metric tracking
            for metric in metrics:
                score_a = result["scores"][label_a][metric]["score"]
                score_b = result["scores"][label_b][metric]["score"]
                
                metric_totals_a[metric] += score_a
                metric_totals_b[metric] += score_b
                
                if score_a > score_b:
                    metric_wins_a[metric] += 1
                elif score_b > score_a:
                    metric_wins_b[metric] += 1
            
        except Exception as e:
            print(f"Error processing comparison {i}: {e}")
            results.append({"error": str(e), "question_index": i})
    
    total = len(questions)
    successful = total - sum(1 for r in results if "error" in r)
    
    return {
        "individual_results": results,
        "summary": {
            "total_comparisons": total,
            "successful_comparisons": successful,
            "overall_winners": {
                f"{label_a}_wins": wins_a,
                f"{label_b}_wins": wins_b,
                "ties": ties,
                f"{label_a}_win_rate": wins_a / successful if successful > 0 else 0,
                f"{label_b}_win_rate": wins_b / successful if successful > 0 else 0
            },
            "score_aggregates": {
                label_a: {
                    "total_points": total_score_a,
                    "average_per_comparison": round(total_score_a / successful, 2) if successful > 0 else 0,
                    "average_per_metric": round(total_score_a / (successful * len(metrics)), 2) if successful > 0 else 0
                },
                label_b: {
                    "total_points": total_score_b,
                    "average_per_comparison": round(total_score_b / successful, 2) if successful > 0 else 0,
                    "average_per_metric": round(total_score_b / (successful * len(metrics)), 2) if successful > 0 else 0
                }
            },
            "metric_breakdown": {
                f"{label_a}_averages": {
                    m: round(metric_totals_a[m] / successful, 2) if successful > 0 else 0 
                    for m in metrics
                },
                f"{label_b}_averages": {
                    m: round(metric_totals_b[m] / successful, 2) if successful > 0 else 0 
                    for m in metrics
                },
                f"{label_a}_metric_wins": metric_wins_a,
                f"{label_b}_metric_wins": metric_wins_b
            }
        }
    }