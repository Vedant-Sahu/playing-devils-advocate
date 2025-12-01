"""
Evaluation pipeline comparing zero-shot, baseline, and adaptive explanation systems.

Runs all three approaches on the same quiz questions and saves:
- Quiz performance metrics for each system
- Final explanations for baseline and adaptive (for three-way judge comparison)
"""

from __future__ import annotations
import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from src.utils.gpqa_sampler import create_gpqa_quiz
from src.graphs.single_adaptive_graph import create_single_adaptive_graph, create_initial_state as single_adaptive_state
from src.graphs.multi_adaptive_graph import create_multi_adaptive_graph, create_initial_state as multi_adaptive_state
from src.graphs.baseline_graph import create_baseline_graph, create_initial_state as baseline_state
from src.agents.multi_metric_judge_agent import batch_three_way_comparison
from src.config.agent_config import _llm, PERSONAS


class EvaluationPipeline:
    """Pipeline for comparing zero-shot, baseline, single adaptive, and multi-adaptive systems."""
    
    def __init__(
        self,
        subset: str = "gpqa_main",
        domain: str = "Physics",
        n_questions: int = 10,
        seed: int = 53,
        results_dir: Path = Path("results"),
        custom_cache_file: str = None
    ):
        self.subset = subset
        self.domain = domain
        self.n_questions = n_questions
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.custom_cache_file = custom_cache_file
        
        self.rng = random.Random(seed)
        
    def generate_quiz(self) -> tuple[List[Dict[str, Any]], List[int]]:
        """Generate quiz questions from GPQA dataset or custom cache file."""
        quiz, indices = create_gpqa_quiz(
            subset=self.subset,
            domain=self.domain,
            seed=self.seed,
            num_questions=self.n_questions,
            custom_cache_file=self.custom_cache_file
        )
        return quiz, indices
    
    def run_zero_shot(self, gpqa_question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run zero-shot: LLM answers quiz directly without explanation.
        
        Returns quiz performance metrics only (no explanation).
        """
        print(f"  Running zero-shot...")
        
        # Direct answer without explanation
        llm = _llm(temperature=0.7, role="zero_shot_answerer")
        
        question = gpqa_question["question"]
        options = gpqa_question["options"]
        correct = gpqa_question["correct"]
        
        prompt = f"""Answer this physics question by selecting the best option.

        Question: {question}

        Options:
        {chr(10).join(options)}

        Provide ONLY the letter of your answer (A, B, C, or D) and a brief one-sentence explanation.

        Format your response as:
        Answer: [LETTER]
        Explanation: [One sentence]"""
        
        response = llm.invoke(prompt).content
        
        # Parse response
        answer_line = [line for line in response.split('\n') if 'Answer:' in line]
        explanation_line = [line for line in response.split('\n') if 'Explanation:' in line]
        
        predicted = answer_line[0].split('Answer:')[-1].strip()[0] if answer_line else "?"
        explanation = explanation_line[0].split('Explanation:')[-1].strip() if explanation_line else ""
        
        is_correct = predicted == correct
        
        return {
            "quiz_results": {
                "total_questions": 1,
                "question_id": gpqa_question["id"],
                "predicted": predicted,
                "is_correct": is_correct,
                "answer_reasoning": explanation,
            },
            "overall_score": 1.0 if is_correct else 0.0,
            "teacher_explanation": None  # No detailed explanation for zero-shot
        }
    
    def run_baseline(
        self, 
        gpqa_question: Dict[str, Any],
        baseline_graph,
    ) -> Dict[str, Any]:
        """
        Run baseline: Single teacher explanation (zero-shot) → grading.
        
        Returns quiz performance metrics and explanation.
        """
        print(f"  Running baseline...")
        
        # Run baseline graph
        baseline_results = baseline_graph.invoke(
            baseline_state(gpqa_question),
            config={"recursion_limit": 30}
        )

        quiz_results = baseline_results.get("quiz_results", {})
        is_correct = quiz_results.get("is_correct", False)
        
        result = {
            "quiz_results": quiz_results,
            "overall_score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "teacher_explanation": baseline_results.get("explanation", ""),
            "single_answer": baseline_results.get("single_answer", ""),
            "answer_reasoning": baseline_results.get("single_explanation", "")
        }
        # Include tool_calls only if present (tools were enabled and used)
        if baseline_results.get("tool_calls"):
            result["tool_calls"] = baseline_results["tool_calls"]
        return result
    
    def run_single_adaptive(
        self,
        gpqa_question: Dict[str, Any],
        single_adaptive_graph,
        max_iters: int = 3
    ) -> Dict[str, Any]:
        """
        Run single adaptive: Single-student refinement → grading.
        
        Returns quiz performance metrics and final explanation.
        """
        print(f"  Running single adaptive (max {max_iters} iterations)...")
        
        # Run single adaptive graph
        single_results = single_adaptive_graph.invoke(
            single_adaptive_state(gpqa_question, max_iters=max_iters),
            config={"recursion_limit": 30}
        )

        quiz_results = single_results.get("quiz_results", {})
        is_correct = quiz_results.get("is_correct", False)
        
        result = {
            "quiz_results": quiz_results,
            "overall_score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "teacher_explanation": single_results.get("explanation", ""),
            "single_answer": single_results.get("single_answer", ""),
            "answer_reasoning": single_results.get("single_explanation", ""),
            "iterations": single_results.get("iteration", 0),
            "critique_history": single_results.get("critique_history", [])
        }
        # Include tool_calls only if present (tools were enabled and used)
        if single_results.get("tool_calls"):
            result["tool_calls"] = single_results["tool_calls"]
        return result
    
    def run_multi_adaptive(
        self,
        gpqa_question: Dict[str, Any],
        multi_adaptive_graph,
        max_iters: int = 3
    ) -> Dict[str, Any]:
        """
        Run multi-adaptive: Full multi-student refinement → grading.
        
        Returns quiz performance metrics and final explanation.
        """
        print(f"  Running multi-adaptive (max {max_iters} iterations)...")
        
        # Run multi-adaptive graph
        multi_results = multi_adaptive_graph.invoke(
            multi_adaptive_state(gpqa_question, max_iters=max_iters),
            config={"recursion_limit": 30}
        )

        quiz_results = multi_results.get("quiz_results", {})
        is_correct = quiz_results.get("is_correct", False)
        
        result = {
            "quiz_results": quiz_results,
            "overall_score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "teacher_explanation": multi_results.get("explanation", ""),
            "single_answer": multi_results.get("single_answer", ""),
            "answer_reasoning": multi_results.get("single_explanation", ""),
            "iterations": multi_results.get("iteration", 0),
            "final_scores": multi_results.get("reward_scores", {}),
            "history": multi_results.get("history", [])
        }
        # Include tool_calls only if present (tools were enabled and used)
        if multi_results.get("tool_calls"):
            result["tool_calls"] = multi_results["tool_calls"]
        return result
    
    def run_full_evaluation(
        self, 
        baseline_graph=None,
        single_adaptive_graph=None,
        multi_adaptive_graph=None,
        max_iters: int = 3,
        run_three_way_judgment: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation: zero-shot, baseline, single adaptive, and multi-adaptive on same quiz.
        
        Args:
            baseline_graph: Pre-built baseline graph (or will create if None)
            single_adaptive_graph: Pre-built single adaptive graph (or will create if None)
            multi_adaptive_graph: Pre-built multi-adaptive graph (or will create if None)
            max_iters: Max iterations for adaptive refinement
            run_three_way_judgment: Whether to run multi-metric judge comparison
            
        Returns:
            Complete evaluation results with all metrics and explanations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*80}")
        print(f"Starting Evaluation Run: {timestamp}")
        print(f"{'='*80}")
        print(f"Dataset: {self.subset} - {self.domain}")
        print(f"Questions: {self.n_questions}")
        print(f"Seed: {self.seed}\n")

        # Create graphs if not provided
        if baseline_graph is None:
            print("Creating baseline graph...")
            baseline_graph = create_baseline_graph()
        
        if single_adaptive_graph is None:
            print("Creating single adaptive graph...")
            single_adaptive_graph = create_single_adaptive_graph()
        
        if multi_adaptive_graph is None:
            print("Creating multi-adaptive graph...")
            multi_adaptive_graph = create_multi_adaptive_graph()
        
        # Generate quiz
        print("Generating quiz...")
        quiz, indices = self.generate_quiz()
        print(f"Generated {len(quiz)} questions\n")
        
        # Store results for each question
        question_results = []
        
        for i, gpqa_question in enumerate(quiz, 1):

            question = gpqa_question.get("question", "")
            question_id = gpqa_question.get("id", "")
            
            print(f"\n{'-'*80}")
            print(f"Question {i}/{len(quiz)} (ID: {question_id})")
            print(f"{'-'*80}")
            print(f"{question[:100]}...")
            
            try:
                # Run all four approaches
                zero_shot_result = self.run_zero_shot(gpqa_question)
                baseline_result = self.run_baseline(gpqa_question, baseline_graph)
                single_adaptive_result = self.run_single_adaptive(gpqa_question, single_adaptive_graph, max_iters)
                multi_adaptive_result = self.run_multi_adaptive(gpqa_question, multi_adaptive_graph, max_iters)
                
                # Build baseline dict
                baseline_data = {
                    "quiz_performance": baseline_result["quiz_results"],
                    "overall_score": baseline_result["overall_score"],
                    "predicted": baseline_result.get("single_answer", "?"),
                    "teacher_explanation": baseline_result["teacher_explanation"],
                    "answer_reasoning": baseline_result.get("answer_reasoning", "")
                }
                # Include tool_calls only if present
                if baseline_result.get("tool_calls"):
                    baseline_data["tool_calls"] = baseline_result["tool_calls"]
                
                # Build single adaptive dict
                single_adaptive_data = {
                    "quiz_performance": single_adaptive_result["quiz_results"],
                    "overall_score": single_adaptive_result["overall_score"],
                    "predicted": single_adaptive_result.get("single_answer", "?"),
                    "teacher_explanation": single_adaptive_result["teacher_explanation"],
                    "answer_reasoning": single_adaptive_result.get("answer_reasoning", ""),
                    "iterations": single_adaptive_result.get("iterations", 0),
                    "critique_history": single_adaptive_result.get("critique_history", [])
                }
                # Include tool_calls only if present
                if single_adaptive_result.get("tool_calls"):
                    single_adaptive_data["tool_calls"] = single_adaptive_result["tool_calls"]
                
                # Build multi-adaptive dict
                multi_adaptive_data = {
                    "quiz_performance": multi_adaptive_result["quiz_results"],
                    "overall_score": multi_adaptive_result["overall_score"],
                    "predicted": multi_adaptive_result.get("single_answer", "?"),
                    "teacher_explanation": multi_adaptive_result["teacher_explanation"],
                    "answer_reasoning": multi_adaptive_result.get("answer_reasoning", ""),
                    "iterations": multi_adaptive_result.get("iterations", 0),
                    "final_scores": multi_adaptive_result.get("final_scores", {}),
                    "history": multi_adaptive_result.get("history", [])
                }
                # Include tool_calls only if present
                if multi_adaptive_result.get("tool_calls"):
                    multi_adaptive_data["tool_calls"] = multi_adaptive_result["tool_calls"]
                
                question_results.append({
                    "question_id": question_id,
                    "question": question,
                    "gpqa_index": indices[i-1] if i-1 < len(indices) else None,
                    "correct_answer": gpqa_question["correct"],
                    "expert_explanation": gpqa_question.get("expert_explanation", ""),
                    "zero_shot": {
                        "quiz_performance": zero_shot_result["quiz_results"],
                        "overall_score": zero_shot_result["overall_score"],
                        "predicted": zero_shot_result["quiz_results"].get("predicted", "?")
                    },
                    "baseline": baseline_data,
                    "single_adaptive": single_adaptive_data,
                    "multi_adaptive": multi_adaptive_data
                })
                
                print(f"\n  Results:")
                print(f"    Zero-shot:       {'✓' if zero_shot_result['overall_score'] == 1.0 else '✗'} (answer: {zero_shot_result['quiz_results'].get('predicted', '?')})")
                print(f"    Baseline:        {'✓' if baseline_result['overall_score'] == 1.0 else '✗'} (answer: {baseline_result.get('single_answer', '?')})")
                print(f"    Single Adaptive: {'✓' if single_adaptive_result['overall_score'] == 1.0 else '✗'} (answer: {single_adaptive_result.get('single_answer', '?')}, {single_adaptive_result.get('iterations', 0)} iterations)")
                print(f"    Multi-Adaptive:  {'✓' if multi_adaptive_result['overall_score'] == 1.0 else '✗'} (answer: {multi_adaptive_result.get('single_answer', '?')}, {multi_adaptive_result.get('iterations', 0)} iterations)")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                question_results.append({
                    "question_id": question_id,
                    "question": question,
                    "error": str(e)
                })
        
        # Aggregate statistics
        summary = self._compute_summary(question_results)
        
        # Save results
        results = {
            "timestamp": timestamp,
            "config": {
                "subset": self.subset,
                "domain": self.domain,
                "n_questions": self.n_questions,
                "seed": self.seed,
                "max_iters": max_iters
            },
            "quiz": quiz,
            "question_results": question_results,
            "summary": summary
        }
        
        output_file = self.results_dir / f"eval_{timestamp}.json"
        output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Run three-way judgment if requested
        if run_three_way_judgment:
            judge_results = self._three_way_judgment(question_results, timestamp)
            results["three_way_judgment"] = judge_results
        
        # Print summary
        self._print_summary(summary)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return results
    
    def _compute_summary(self, question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics across all questions."""
        valid_results = [r for r in question_results if "error" not in r]
        n_valid = len(valid_results)
        
        if n_valid == 0:
            return {"error": "No valid results"}
        
        # Count correct answers
        zero_shot_correct = sum(1 for r in valid_results if r["zero_shot"]["overall_score"] == 1.0)
        baseline_correct = sum(1 for r in valid_results if r["baseline"]["overall_score"] == 1.0)
        single_adaptive_correct = sum(1 for r in valid_results if r["single_adaptive"]["overall_score"] == 1.0)
        multi_adaptive_correct = sum(1 for r in valid_results if r["multi_adaptive"]["overall_score"] == 1.0)
        
        # Count wins (who got it correct when at least one other was wrong)
        zero_shot_wins = sum(1 for r in valid_results 
                             if r["zero_shot"]["overall_score"] == 1.0
                             and any(r[k]["overall_score"] == 0.0 for k in ["baseline", "single_adaptive", "multi_adaptive"]))
        baseline_wins = sum(1 for r in valid_results 
                           if r["baseline"]["overall_score"] == 1.0
                           and any(r[k]["overall_score"] == 0.0 for k in ["zero_shot", "single_adaptive", "multi_adaptive"]))
        single_adaptive_wins = sum(1 for r in valid_results 
                           if r["single_adaptive"]["overall_score"] == 1.0
                           and any(r[k]["overall_score"] == 0.0 for k in ["zero_shot", "baseline", "multi_adaptive"]))
        multi_adaptive_wins = sum(1 for r in valid_results 
                           if r["multi_adaptive"]["overall_score"] == 1.0
                           and any(r[k]["overall_score"] == 0.0 for k in ["zero_shot", "baseline", "single_adaptive"]))
        
        # Average iterations for adaptive systems
        avg_single_iterations = sum(r["single_adaptive"]["iterations"] for r in valid_results) / n_valid
        avg_multi_iterations = sum(r["multi_adaptive"]["iterations"] for r in valid_results) / n_valid
        
        return {
            "n_questions": n_valid,
            "accuracy": {
                "zero_shot": zero_shot_correct / n_valid,
                "baseline": baseline_correct / n_valid,
                "single_adaptive": single_adaptive_correct / n_valid,
                "multi_adaptive": multi_adaptive_correct / n_valid
            },
            "correct_counts": {
                "zero_shot": zero_shot_correct,
                "baseline": baseline_correct,
                "single_adaptive": single_adaptive_correct,
                "multi_adaptive": multi_adaptive_correct
            },
            "wins": {
                "zero_shot": zero_shot_wins,
                "baseline": baseline_wins,
                "single_adaptive": single_adaptive_wins,
                "multi_adaptive": multi_adaptive_wins
            },
            "adaptive_metrics": {
                "single_average_iterations": avg_single_iterations,
                "multi_average_iterations": avg_multi_iterations
            },
            "improvements": {
                "single_adaptive_vs_zero_shot": (single_adaptive_correct - zero_shot_correct) / n_valid,
                "single_adaptive_vs_baseline": (single_adaptive_correct - baseline_correct) / n_valid,
                "multi_adaptive_vs_zero_shot": (multi_adaptive_correct - zero_shot_correct) / n_valid,
                "multi_adaptive_vs_baseline": (multi_adaptive_correct - baseline_correct) / n_valid,
                "multi_adaptive_vs_single_adaptive": (multi_adaptive_correct - single_adaptive_correct) / n_valid,
                "baseline_vs_zero_shot": (baseline_correct - zero_shot_correct) / n_valid
            }
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary statistics."""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Valid Questions: {summary['n_questions']}")
        
        print(f"\nAccuracy (Correct Answers):")
        print(f"  Zero-shot:       {summary['correct_counts']['zero_shot']}/{summary['n_questions']} ({summary['accuracy']['zero_shot']*100:.1f}%)")
        print(f"  Baseline:        {summary['correct_counts']['baseline']}/{summary['n_questions']} ({summary['accuracy']['baseline']*100:.1f}%)")
        print(f"  Single Adaptive: {summary['correct_counts']['single_adaptive']}/{summary['n_questions']} ({summary['accuracy']['single_adaptive']*100:.1f}%)")
        print(f"  Multi-Adaptive:  {summary['correct_counts']['multi_adaptive']}/{summary['n_questions']} ({summary['accuracy']['multi_adaptive']*100:.1f}%)")
        
        print(f"\nWins (correct when at least one other was wrong):")
        print(f"  Zero-shot:       {summary['wins']['zero_shot']}")
        print(f"  Baseline:        {summary['wins']['baseline']}")
        print(f"  Single Adaptive: {summary['wins']['single_adaptive']}")
        print(f"  Multi-Adaptive:  {summary['wins']['multi_adaptive']}")
        
        print(f"\nImprovements (accuracy difference):")
        print(f"  Single Adaptive vs Zero-shot: {summary['improvements']['single_adaptive_vs_zero_shot']*100:+.1f}%")
        print(f"  Single Adaptive vs Baseline:  {summary['improvements']['single_adaptive_vs_baseline']*100:+.1f}%")
        print(f"  Multi-Adaptive vs Zero-shot:  {summary['improvements']['multi_adaptive_vs_zero_shot']*100:+.1f}%")
        print(f"  Multi-Adaptive vs Baseline:   {summary['improvements']['multi_adaptive_vs_baseline']*100:+.1f}%")
        print(f"  Multi vs Single Adaptive:     {summary['improvements']['multi_adaptive_vs_single_adaptive']*100:+.1f}%")
        print(f"  Baseline vs Zero-shot:        {summary['improvements']['baseline_vs_zero_shot']*100:+.1f}%")
        
        print(f"\nAdaptive Systems:")
        print(f"  Single Adaptive Avg Iterations: {summary['adaptive_metrics']['single_average_iterations']:.1f}")
        print(f"  Multi-Adaptive Avg Iterations:  {summary['adaptive_metrics']['multi_average_iterations']:.1f}")


    def _three_way_judgment(
            self, 
            question_results: List[Dict[str, Any]], 
            timestamp: str
        ) -> Dict[str, Any]:
        """
        Run three-way comparison between baseline, single adaptive, and multi-adaptive using unified judge.
        
        Args:
            question_results: List of question results from evaluation
            timestamp: Timestamp string for output file naming
            
        Returns:
            Dictionary with comprehensive three-way comparison results including rankings
        """
        from src.agents.multi_metric_judge_agent import batch_three_way_comparison
        
        # Extract data for three-way comparison
        questions = []
        correct_answers = []
        expert_explanations = []
        baseline_explanations = []
        single_adaptive_explanations = []
        multi_adaptive_explanations = []
        metadata = []
        
        for qr in question_results:
            if "error" in qr:
                continue
            
            questions.append(qr["question"])
            correct_answers.append(qr["correct_answer"])  
            expert_explanations.append(qr.get("expert_explanation", ""))
            baseline_explanations.append(qr["baseline"]["teacher_explanation"])
            single_adaptive_explanations.append(qr["single_adaptive"]["teacher_explanation"])
            multi_adaptive_explanations.append(qr["multi_adaptive"]["teacher_explanation"])
            
            metadata.append({
                "question_id": qr["question_id"],
                "baseline_score": qr["baseline"]["overall_score"],
                "single_adaptive_score": qr["single_adaptive"]["overall_score"],
                "multi_adaptive_score": qr["multi_adaptive"]["overall_score"],
                "baseline_correct": qr["baseline"]["overall_score"] == 1.0,
                "single_adaptive_correct": qr["single_adaptive"]["overall_score"] == 1.0,
                "multi_adaptive_correct": qr["multi_adaptive"]["overall_score"] == 1.0,
                "correct_answer": qr["correct_answer"],
                "baseline_predicted": qr["baseline"]["predicted"],
                "single_adaptive_predicted": qr["single_adaptive"]["predicted"],
                "multi_adaptive_predicted": qr["multi_adaptive"]["predicted"]
            })
        
        if len(questions) == 0:
            return {
                "error": "No valid question results found",
                "summary": {}
            }
        
        print(f"\n{'='*80}")
        print("THREE-WAY SOLUTION GUIDE EVALUATION")
        print(f"{'='*80}")
        print(f"Evaluating: Baseline vs Single Adaptive vs Multi-Adaptive")
        print(f"Questions: {len(questions)}")
        print(f"Metrics: Solution Correctness, Step-by-Step Clarity, Completeness,")
        print(f"         Mathematical Precision, Conceptual Grounding, Graduate-Level Appropriateness\n")
        
        # Run unified three-way comparison
        judge_results = batch_three_way_comparison(
            questions=questions,
            correct_answers=correct_answers,
            baseline_explanations=baseline_explanations,
            single_adaptive_explanations=single_adaptive_explanations,
            multi_adaptive_explanations=multi_adaptive_explanations,
            expert_explanations=expert_explanations
        )
        
        # Add metadata to individual results
        for i, meta in enumerate(metadata):
            if i < len(judge_results["individual_results"]):
                if "error" not in judge_results["individual_results"][i]:
                    judge_results["individual_results"][i]["metadata"] = meta
        
        # Print comprehensive summary
        summary = judge_results["summary"]
        print(f"\n{'='*80}")
        print("THREE-WAY COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Total Comparisons: {summary['total_comparisons']}")
        print(f"Successful Comparisons: {summary['successful_comparisons']}\n")
        
        # Print overall placements
        print("OVERALL PLACEMENTS (across all questions):")
        placements = summary['overall_placements']
        for system in ["baseline", "single_adaptive", "multi_adaptive"]:
            first = placements[system]["1st"]
            second = placements[system]["2nd"]
            third = placements[system]["3rd"]
            avg_score = summary['average_scores'][system]
            print(f"  {system.replace('_', ' ').title()}:")
            print(f"    1st place: {first}, 2nd place: {second}, 3rd place: {third}")
            print(f"    Average score: {avg_score}/60")
        
        # Print metric-by-metric breakdown
        print(f"\nMETRIC-BY-METRIC BREAKDOWN (average scores 0-10):")
        metrics = [
            "solution_correctness", "step_by_step_clarity", "completeness",
            "mathematical_precision", "conceptual_grounding", "graduate_level_appropriateness"
        ]
        
        for metric in metrics:
            print(f"\n  {metric.replace('_', ' ').title()}:")
            metric_avgs = summary['metric_averages']
            for system in ["baseline", "single_adaptive", "multi_adaptive"]:
                avg = metric_avgs[system][metric]
                rank_dist = summary['metric_rank_distributions'][system][metric]
                print(f"    {system.replace('_', ' ').title()}: {avg:.2f}/10 "
                      f"(1st: {rank_dist[1]}, 2nd: {rank_dist[2]}, 3rd: {rank_dist[3]})")
        
        # Determine overall winner
        print(f"\n{'='*80}")
        print("OVERALL WINNER:")
        print(f"{'='*80}")
        
        # Rank by 1st place finishes, then average score
        systems_with_stats = []
        for system in ["baseline", "single_adaptive", "multi_adaptive"]:
            first_count = placements[system]["1st"]
            avg_score = summary['average_scores'][system]
            systems_with_stats.append((system, first_count, avg_score))
        
        # Sort by 1st place finishes (desc), then by average score (desc)
        ranked = sorted(systems_with_stats, key=lambda x: (x[1], x[2]), reverse=True)
        
        for rank, (system, firsts, avg) in enumerate(ranked, 1):
            medal = ["🥇", "🥈", "🥉"][rank-1]
            print(f"  {medal} {rank}. {system.replace('_', ' ').title()}: "
                  f"{firsts} first-place finishes, {avg:.2f}/60 avg score")
        
        # Save results
        judge_output = self.results_dir / f"judge_eval_three_way_{timestamp}.json"
        judge_results["metadata"] = metadata
        judge_results["overall_ranking"] = {
            "ranking": [{"system": s, "first_places": f, "avg_score": a} for s, f, a in ranked],
            "placements": placements,
            "average_scores": summary['average_scores']
        }
        judge_output.write_text(
            json.dumps(judge_results, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"\n{'='*80}")
        print(f"Results saved to: {judge_output}")
        print(f"{'='*80}\n")
        
        return judge_results