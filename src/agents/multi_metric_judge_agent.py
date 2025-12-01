from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.agent_config import _llm
from src.utils.parsing import _extract_json


def three_way_judge_solution_guides(
    question: str,
    correct_answer: str,
    baseline_explanation: str,
    single_adaptive_explanation: str,
    multi_adaptive_explanation: str,
    expert_explanation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate three solution guides (baseline, single adaptive, multi-adaptive) using
    comprehensive physics education metrics. Focus on step-by-step solution guidance.
    
    Args:
        question: The physics question being explained
        correct_answer: The correct answer to the question
        baseline_explanation: Baseline solution guide
        single_adaptive_explanation: Single adaptive solution guide
        multi_adaptive_explanation: Multi-adaptive solution guide
        expert_explanation: Optional reference explanation for additional context
    
    Returns:
        Dict containing per-metric rankings (1-3), scores (0-10), reasoning, and overall rankings
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        raise ValueError("correct_answer must be a non-empty string.")
    if not isinstance(baseline_explanation, str) or not baseline_explanation.strip():
        raise ValueError("baseline_explanation must be a non-empty string.")
    if not isinstance(single_adaptive_explanation, str) or not single_adaptive_explanation.strip():
        raise ValueError("single_adaptive_explanation must be a non-empty string.")
    if not isinstance(multi_adaptive_explanation, str) or not multi_adaptive_explanation.strip():
        raise ValueError("multi_adaptive_explanation must be a non-empty string.")

    llm = _llm(temperature=0.2, json_mode=True, role="three_way_judge", max_tokens=8000)

    # Comprehensive three-way evaluation system focused on solution guides
    sys = SystemMessage(
        content="""You are an expert Physics Education Researcher evaluating SOLUTION GUIDES for graduate-level physics questions.

Your task: Evaluate THREE solution guides across 6 independent metrics. For each metric:
1. First, carefully read all three solution guides
2. Provide detailed chain-of-thought reasoning comparing all three
3. Assign a score from 0-10 (integers only) to EACH guide independently
4. Rank them 1st (best), 2nd (middle), 3rd (worst) for that metric

CRITICAL CONTEXT:
• These are SOLUTION GUIDES, not general explanations - they should provide step-by-step instructions on HOW TO SOLVE this specific problem
• You have access to the CORRECT ANSWER
• Your primary job is to evaluate whether each guide leads students to the correct answer through clear, accurate steps
• Graduate-level physics problems are challenging - scrutinize carefully for missing steps, incorrect reasoning, or misleading guidance

═══════════════════════════════════════════════════════════════════════════════════
EVALUATION METRICS (score and rank each solution guide 0-10 and 1-3):
═══════════════════════════════════════════════════════════════════════════════════

1. **SOLUTION CORRECTNESS** (Does it guide toward the correct answer?)
   
   Scoring Guide (0-10 integers):
   • 9-10: All steps correct, directly guides to correct answer, no errors or red herrings
   • 7-8:  Correct approach with only minor imprecisions that don't affect solving
   • 5-6:  Generally correct but has some unclear or potentially confusing steps
   • 3-4:  Mix of correct and incorrect guidance, could lead students astray
   • 1-2:  Significant errors in solution steps that mislead toward wrong answer
   • 0:     Fundamentally incorrect approach or contradicts correct solution path
   
   Critical evaluation points:
   - Does each step logically lead toward the CORRECT ANSWER?
   - Are the recommended equations/formulas correct for this problem?
   - Are there any errors in the solution approach?
   - Would a student following these steps arrive at the correct answer?
   - Are there any red herrings or misleading suggestions?

2. **STEP-BY-STEP CLARITY** (How clear is the solution procedure?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Exceptional - each step clearly defined, logical order, explains what to do and why
   • 7-8:  Clear step sequence with good explanations of procedure
   • 5-6:  Steps present but could be more explicit or better ordered
   • 3-4:  Some steps vague or confusing, unclear ordering
   • 1-2:  Very unclear procedure, missing critical steps
   • 0:     No discernible step-by-step structure
   
   Critical evaluation points:
   - Are the steps presented in a logical, executable order?
   - Is each step clearly defined (what to do, not just concepts to know)?
   - Are there gaps between steps that would confuse students?
   - Does it explain HOW to apply formulas, not just which ones to use?
   - Can a student follow this as an actionable procedure?

3. **COMPLETENESS** (Are all necessary solution steps covered?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Comprehensive - all critical steps from problem setup to final answer
   • 7-8:  Covers key steps well, only minor omissions that students could infer
   • 5-6:  Main steps present but missing some important intermediate steps
   • 3-4:  Significant gaps - missing steps that would leave students stuck
   • 1-2:  Many critical steps missing, incomplete solution path
   • 0:     Severely incomplete, barely addresses how to solve
   
   Critical evaluation points:
   - Are ALL steps needed to solve this problem included?
   - Is problem setup/variable identification covered?
   - Are intermediate calculations explained?
   - Is the method for combining results described?
   - Would students know how to get from start to finish?

4. **MATHEMATICAL PRECISION** (Are formulas and calculations accurate?)
   
   Scoring Guide (0-10 integers):
   • 9-10: All equations correct, notation clear, mathematical reasoning sound
   • 7-8:  Math generally correct, minor notation issues that don't cause confusion
   • 5-6:  Mostly correct but some imprecise notation or unclear variable usage
   • 3-4:  Several mathematical errors or confusing notation
   • 1-2:  Significant math errors that would lead to wrong answer
   • 0:     Mathematics fundamentally wrong
   
   Critical evaluation points:
   - Are all recommended formulas/equations correct for this problem?
   - Is mathematical notation used consistently and correctly?
   - Are variables clearly defined?
   - Are unit considerations mentioned when important?
   - Would the mathematical approach work for this specific problem?

5. **CONCEPTUAL GROUNDING** (Does it explain the physics WHY behind steps?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Excellent - explains physics reasoning for each step, builds understanding
   • 7-8:  Good conceptual explanations supporting the procedure
   • 5-6:  Some physics reasoning but could be more explanatory
   • 3-4:  Mostly procedural, limited physics reasoning
   • 1-2:  Almost no conceptual grounding, just formula listing
   • 0:     No physics reasoning, mechanical steps only
   
   Critical evaluation points:
   - Does it explain WHY each step is necessary (physics reasoning)?
   - Are relevant physical principles identified?
   - Does it connect steps to underlying physics concepts?
   - Would students understand the physics, not just the procedure?
   - Are key physics insights (conservation laws, symmetries, etc.) explained?

6. **GRADUATE-LEVEL APPROPRIATENESS** (Is it pitched correctly?)
   
   Scoring Guide (0-10 integers):
   • 9-10: Perfectly calibrated - rigorous yet clear, appropriate for graduate students
   • 7-8:  Appropriate level with good balance of rigor and accessibility
   • 5-6:  Generally appropriate but some sections too simple or too advanced
   • 3-4:  Often mismatched (too elementary or too terse for graduate level)
   • 1-2:  Significantly inappropriate for graduate physics students
   • 0:     Completely wrong level
   
   Critical evaluation points:
   - Does it assume appropriate prerequisite knowledge?
   - Is the level of mathematical detail appropriate?
   - Are explanations neither too basic nor too advanced?
   - Would graduate physics students find this helpful and appropriately challenging?

═══════════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (JSON only):
═══════════════════════════════════════════════════════════════════════════════════

{
  "baseline_scores": {
    "solution_correctness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence detailed evaluation. Cite specific steps or issues.>"
    },
    "step_by_step_clarity": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation with specific examples>"
    },
    "completeness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation noting what's present or missing>"
    },
    "mathematical_precision": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation of math accuracy>"
    },
    "conceptual_grounding": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation of physics reasoning>"
    },
    "graduate_level_appropriateness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation of level appropriateness>"
    }
  },
  "single_adaptive_scores": {
    "solution_correctness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence detailed evaluation>"
    },
    "step_by_step_clarity": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "completeness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "mathematical_precision": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "conceptual_grounding": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "graduate_level_appropriateness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    }
  },
  "multi_adaptive_scores": {
    "solution_correctness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence detailed evaluation>"
    },
    "step_by_step_clarity": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "completeness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "mathematical_precision": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "conceptual_grounding": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    },
    "graduate_level_appropriateness": {
      "score": <integer 0-10>,
      "rank": <integer 1-3>,
      "reasoning": "<3-4 sentence evaluation>"
    }
  },
  "overall_analysis": {
    "overall_ranking": "<'baseline', 'single_adaptive', or 'multi_adaptive' - which guide is BEST overall>",
    "second_place": "<which is second best>",
    "third_place": "<which is third/worst>",
    "summary": "<3-4 sentence summary comparing all three guides and explaining the ranking>"
  }
}

CRITICAL REQUIREMENTS:
• Each guide must be ranked 1, 2, or 3 for EACH metric (no ties - if scores are equal, use reasoning to break tie)
• Reasoning must be SPECIFIC - cite actual steps, formulas, or issues from each guide
• Focus evaluation on whether guides lead to CORRECT ANSWER with clear, complete steps
• Be rigorous - these are graduate-level physics problems requiring careful solution procedures

Return ONLY valid JSON."""
    )

    # Build human message with all three explanations
    expert_section = f"\n\nEXPERT REFERENCE (optional context):\n{expert_explanation}\n" if expert_explanation else ""
    
    hum = HumanMessage(
        content=(
            f"PHYSICS QUESTION:\n{question}\n\n"
            f"CORRECT ANSWER: {correct_answer}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"BASELINE SOLUTION GUIDE:\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"{baseline_explanation}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"SINGLE ADAPTIVE SOLUTION GUIDE:\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"{single_adaptive_explanation}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"MULTI-ADAPTIVE SOLUTION GUIDE:\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"{multi_adaptive_explanation}\n"
            f"{expert_section}\n"
            f"Evaluate all three solution guides. Return JSON only with scores (0-10) and rankings (1-3) for each metric."
        )
    )

    resp = llm.invoke([sys, hum])
    raw = resp.content
    parsed = raw if isinstance(raw, dict) else _extract_json(raw if isinstance(raw, str) else str(raw))
    
    if not isinstance(parsed, dict):
        raise ValueError("Three-way Judge must return a JSON object.")

    # Validate response structure
    for key in ["baseline_scores", "single_adaptive_scores", "multi_adaptive_scores", "overall_analysis"]:
        if key not in parsed:
            raise ValueError(f"Response must include '{key}'")

    # Define metrics
    metrics = [
        "solution_correctness", "step_by_step_clarity", "completeness",
        "mathematical_precision", "conceptual_grounding", "graduate_level_appropriateness"
    ]
    
    systems = ["baseline", "single_adaptive", "multi_adaptive"]
    scores_by_system = {
        "baseline": parsed["baseline_scores"],
        "single_adaptive": parsed["single_adaptive_scores"],
        "multi_adaptive": parsed["multi_adaptive_scores"]
    }
    
    # Validate all metrics present and scores/ranks are valid
    for system in systems:
        sys_scores = scores_by_system[system]
        for metric in metrics:
            if metric not in sys_scores:
                raise ValueError(f"Missing metric '{metric}' for {system}")
            score = sys_scores[metric].get("score")
            rank = sys_scores[metric].get("rank")
            if not isinstance(score, int) or not (0 <= score <= 10):
                raise ValueError(f"Score for {metric} ({system}) must be integer 0-10, got {score}")
            if not isinstance(rank, int) or not (1 <= rank <= 3):
                raise ValueError(f"Rank for {metric} ({system}) must be integer 1-3, got {rank}")

    # Calculate totals and averages
    totals = {}
    averages = {}
    for system in systems:
        total = sum(scores_by_system[system][m]["score"] for m in metrics)
        totals[system] = total
        averages[system] = round(total / len(metrics), 2)
    
    # Count rankings (how many 1st, 2nd, 3rd place finishes for each system)
    rank_counts = {system: {1: 0, 2: 0, 3: 0} for system in systems}
    for metric in metrics:
        for system in systems:
            rank = scores_by_system[system][metric]["rank"]
            rank_counts[system][rank] += 1
    
    # Verify overall ranking matches the data
    overall_ranking = parsed["overall_analysis"].get("overall_ranking", "").strip().lower()
    second_place = parsed["overall_analysis"].get("second_place", "").strip().lower()
    third_place = parsed["overall_analysis"].get("third_place", "").strip().lower()
    
    # Validate ranking names
    valid_systems = {"baseline", "single_adaptive", "multi_adaptive"}
    if overall_ranking not in valid_systems or second_place not in valid_systems or third_place not in valid_systems:
        # Auto-determine based on totals
        ranked = sorted(systems, key=lambda s: totals[s], reverse=True)
        overall_ranking, second_place, third_place = ranked[0], ranked[1], ranked[2]
        print(f"Warning: Invalid ranking, auto-determined as {overall_ranking} > {second_place} > {third_place}")

    return {
        "scores_by_system": scores_by_system,
        "aggregates": {
            system: {
                "total_score": totals[system],
                "average_score": averages[system],
                "max_possible": len(metrics) * 10,
                "rank_distribution": rank_counts[system]
            }
            for system in systems
        },
        "rankings": {
            "1st_place": overall_ranking,
            "2nd_place": second_place,
            "3rd_place": third_place
        },
        "rank_counts": rank_counts,
        "overall_analysis": parsed["overall_analysis"],
        "metrics_evaluated": metrics
    }


def batch_three_way_comparison(
    questions: List[str],
    correct_answers: List[str],
    baseline_explanations: List[str],
    single_adaptive_explanations: List[str],
    multi_adaptive_explanations: List[str],
    expert_explanations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run three-way comparisons on multiple question-explanation triplets.
    
    Returns aggregate statistics tracking how often each system ranks 1st, 2nd, 3rd.
    """
    n = len(questions)
    if not (n == len(correct_answers) == len(baseline_explanations) == 
            len(single_adaptive_explanations) == len(multi_adaptive_explanations)):
        raise ValueError("All input lists must have the same length")
    
    if expert_explanations is not None and len(expert_explanations) != n:
        raise ValueError("expert_explanations must match length of questions")

    results = []
    systems = ["baseline", "single_adaptive", "multi_adaptive"]
    
    # Track overall rankings across all questions
    place_counts = {
        "baseline": {"1st": 0, "2nd": 0, "3rd": 0},
        "single_adaptive": {"1st": 0, "2nd": 0, "3rd": 0},
        "multi_adaptive": {"1st": 0, "2nd": 0, "3rd": 0}
    }
    
    # Track per-metric rank counts
    metrics = [
        "solution_correctness", "step_by_step_clarity", "completeness",
        "mathematical_precision", "conceptual_grounding", "graduate_level_appropriateness"
    ]
    
    metric_rank_counts = {
        system: {metric: {1: 0, 2: 0, 3: 0} for metric in metrics}
        for system in systems
    }
    
    # Track scores for averaging
    total_scores = {system: 0 for system in systems}
    metric_score_totals = {system: {metric: 0 for metric in metrics} for system in systems}
    
    for i in range(n):
        try:
            expert = expert_explanations[i] if expert_explanations else None
            result = three_way_judge_solution_guides(
                questions[i],
                correct_answers[i],
                baseline_explanations[i],
                single_adaptive_explanations[i],
                multi_adaptive_explanations[i],
                expert
            )
            results.append(result)
            
            # Count overall placements
            place_counts[result["rankings"]["1st_place"]]["1st"] += 1
            place_counts[result["rankings"]["2nd_place"]]["2nd"] += 1
            place_counts[result["rankings"]["3rd_place"]]["3rd"] += 1
            
            # Aggregate scores
            for system in systems:
                total_scores[system] += result["aggregates"][system]["total_score"]
                
                # Per-metric tracking
                for metric in metrics:
                    score = result["scores_by_system"][system][metric]["score"]
                    rank = result["scores_by_system"][system][metric]["rank"]
                    
                    metric_score_totals[system][metric] += score
                    metric_rank_counts[system][metric][rank] += 1
            
        except Exception as e:
            print(f"Error processing comparison {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"error": str(e), "question_index": i})
    
    successful = n - sum(1 for r in results if "error" in r)
    
    # Calculate averages
    avg_scores = {
        system: round(total_scores[system] / successful, 2) if successful > 0 else 0
        for system in systems
    }
    
    metric_avg_scores = {
        system: {
            metric: round(metric_score_totals[system][metric] / successful, 2) if successful > 0 else 0
            for metric in metrics
        }
        for system in systems
    }
    
    return {
        "individual_results": results,
        "summary": {
            "total_comparisons": n,
            "successful_comparisons": successful,
            "overall_placements": place_counts,
            "average_scores": avg_scores,
            "metric_averages": metric_avg_scores,
            "metric_rank_distributions": metric_rank_counts
        }
    }