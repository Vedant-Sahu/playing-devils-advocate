"""
Evaluation pipeline comparing zero-shot, baseline, and adaptive explanation systems.

Runs all three approaches on the same quiz questions and saves:
- Quiz performance metrics for each system
- Final explanations for baseline and adaptive (for pairwise judge comparison)
"""

from __future__ import annotations
import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from src.utils.gpqa_sampler import create_gpqa_quiz
from src.graphs.adaptive_refinement_graph import create_adaptive_refinement_graph, create_initial_state as adaptive_state
from src.graphs.baseline_graph import create_baseline_graph, create_initial_state as baseline_state
from src.agents.multi_metric_judge_agent import batch_multi_metric_comparison
from src.config.agent_config import _llm, PERSONAS


class EvaluationPipeline:
    """Pipeline for comparing zero-shot, baseline, and adaptive systems."""
    
    def __init__(
        self,
        subset: str = "gpqa_main",
        domain: str = "Physics",
        n_questions: int = 10,
        seed: int = 53,
        results_dir: Path = Path("results")
    ):
        self.subset = subset
        self.domain = domain
        self.n_questions = n_questions
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.rng = random.Random(seed)
        
    def generate_quiz(self) -> tuple[List[Dict[str, Any]], List[int]]:
        """Generate quiz questions from GPQA dataset."""
        quiz, indices = create_gpqa_quiz(
            subset=self.subset,
            domain=self.domain,
            seed=self.seed,
            num_questions=self.n_questions
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
        
        return {
            "quiz_results": quiz_results,
            "overall_score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "teacher_explanation": baseline_results.get("explanation", ""),
            "single_answer": baseline_results.get("single_answer", ""),
            "answer_reasoning": baseline_results.get("single_explanation", "")
        }
    
    def run_adaptive(
        self,
        gpqa_question: Dict[str, Any],
        adaptive_graph,
        max_iters: int = 3
    ) -> Dict[str, Any]:
        """
        Run adaptive: Full multi-agent refinement → grading.
        
        Returns quiz performance metrics and final explanation.
        """
        print(f"  Running adaptive (max {max_iters} iterations)...")
        
        # Run adaptive graph
        adaptive_results = adaptive_graph.invoke(
            adaptive_state(gpqa_question, max_iters=max_iters),
            config={"recursion_limit": 30}
        )

        quiz_results = adaptive_results.get("quiz_results", {})
        is_correct = quiz_results.get("is_correct", False)
        
        return {
            "quiz_results": quiz_results,
            "overall_score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "teacher_explanation": adaptive_results.get("explanation", ""),
            "single_answer": adaptive_results.get("single_answer", ""),
            "answer_reasoning": adaptive_results.get("single_explanation", ""),
            "iterations": adaptive_results.get("iteration", 0),
            "final_scores": adaptive_results.get("reward_scores", {}),
            "history": adaptive_results.get("history", [])
        }
    
    def run_full_evaluation(
        self, 
        baseline_graph=None,
        adaptive_graph=None,
        max_iters: int = 3,
        run_pairwise_judgment: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation: zero-shot, baseline, and adaptive on same quiz.
        
        Args:
            baseline_graph: Pre-built baseline graph (or will create if None)
            adaptive_graph: Pre-built adaptive graph (or will create if None)
            max_iters: Max iterations for adaptive refinement
            
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
        
        if adaptive_graph is None:
            print("Creating adaptive graph...")
            adaptive_graph = create_adaptive_refinement_graph()
        
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
                # Run all three approaches
                zero_shot_result = self.run_zero_shot(gpqa_question)
                baseline_result = self.run_baseline(gpqa_question, baseline_graph)
                adaptive_result = self.run_adaptive(gpqa_question, adaptive_graph, max_iters)
                
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
                    "baseline": {
                        "quiz_performance": baseline_result["quiz_results"],
                        "overall_score": baseline_result["overall_score"],
                        "predicted": baseline_result.get("single_answer", "?"),
                        "teacher_explanation": baseline_result["teacher_explanation"],
                        "answer_reasoning": baseline_result.get("answer_reasoning", "")
                    },
                    "adaptive": {
                        "quiz_performance": adaptive_result["quiz_results"],
                        "overall_score": adaptive_result["overall_score"],
                        "predicted": adaptive_result.get("single_answer", "?"),
                        "teacher_explanation": adaptive_result["teacher_explanation"],
                        "answer_reasoning": adaptive_result.get("answer_reasoning", ""),
                        "iterations": adaptive_result.get("iterations", 0),
                        "final_scores": adaptive_result.get("final_scores", {})
                    }
                })
                
                print(f"\n  Results:")
                print(f"    Zero-shot: {'✓' if zero_shot_result['overall_score'] == 1.0 else '✗'} (answer: {zero_shot_result['quiz_results'].get('predicted', '?')})")
                print(f"    Baseline:  {'✓' if baseline_result['overall_score'] == 1.0 else '✗'} (answer: {baseline_result.get('single_answer', '?')})")
                print(f"    Adaptive:  {'✓' if adaptive_result['overall_score'] == 1.0 else '✗'} (answer: {adaptive_result.get('single_answer', '?')}, {adaptive_result.get('iterations', 0)} iterations)")
                
                
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

        # Run pairwise judgment if requested
        if run_pairwise_judgment:
            judge_results = self._pairwise_judgment(question_results, timestamp)
            results["pairwise_judgment"] = judge_results
        
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
        adaptive_correct = sum(1 for r in valid_results if r["adaptive"]["overall_score"] == 1.0)
        
        # Count wins (who got it correct when others didn't)
        zero_shot_wins = sum(1 for r in valid_results 
                             if r["zero_shot"]["overall_score"] == 1.0
                             and (r["baseline"]["overall_score"] == 0.0 or r["adaptive"]["overall_score"] == 0.0))
        baseline_wins = sum(1 for r in valid_results 
                           if r["baseline"]["overall_score"] == 1.0
                           and (r["zero_shot"]["overall_score"] == 0.0 or r["adaptive"]["overall_score"] == 0.0))
        adaptive_wins = sum(1 for r in valid_results 
                           if r["adaptive"]["overall_score"] == 1.0
                           and (r["zero_shot"]["overall_score"] == 0.0 or r["baseline"]["overall_score"] == 0.0))
        
        # Average iterations for adaptive
        avg_iterations = sum(r["adaptive"]["iterations"] for r in valid_results) / n_valid
        
        return {
            "n_questions": n_valid,
            "accuracy": {
                "zero_shot": zero_shot_correct / n_valid,
                "baseline": baseline_correct / n_valid,
                "adaptive": adaptive_correct / n_valid
            },
            "correct_counts": {
                "zero_shot": zero_shot_correct,
                "baseline": baseline_correct,
                "adaptive": adaptive_correct
            },
            "wins": {
                "zero_shot": zero_shot_wins,
                "baseline": baseline_wins,
                "adaptive": adaptive_wins
            },
            "adaptive_metrics": {
                "average_iterations": avg_iterations
            },
            "improvements": {
                "adaptive_vs_zero_shot": (adaptive_correct - zero_shot_correct) / n_valid,
                "adaptive_vs_baseline": (adaptive_correct - baseline_correct) / n_valid,
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
        print(f"  Zero-shot: {summary['correct_counts']['zero_shot']}/{summary['n_questions']} ({summary['accuracy']['zero_shot']*100:.1f}%)")
        print(f"  Baseline:  {summary['correct_counts']['baseline']}/{summary['n_questions']} ({summary['accuracy']['baseline']*100:.1f}%)")
        print(f"  Adaptive:  {summary['correct_counts']['adaptive']}/{summary['n_questions']} ({summary['accuracy']['adaptive']*100:.1f}%)")
        
        print(f"\nWins (correct when at least one other was wrong):")
        print(f"  Zero-shot: {summary['wins']['zero_shot']}")
        print(f"  Baseline:  {summary['wins']['baseline']}")
        print(f"  Adaptive:  {summary['wins']['adaptive']}")
        
        print(f"\nImprovements (accuracy difference):")
        print(f"  Adaptive vs Zero-shot: {summary['improvements']['adaptive_vs_zero_shot']*100:+.1f}%")
        print(f"  Adaptive vs Baseline:  {summary['improvements']['adaptive_vs_baseline']*100:+.1f}%")
        print(f"  Baseline vs Zero-shot: {summary['improvements']['baseline_vs_zero_shot']*100:+.1f}%")
        
        print(f"\nAdaptive System:")
        print(f"  Average Iterations: {summary['adaptive_metrics']['average_iterations']:.1f}")


    def _pairwise_judgment(
            self, 
            question_results: List[Dict[str, Any]], 
            timestamp: str
        ) -> Dict[str, Any]:
        """
        Run multi-metric judge comparison between baseline and adaptive explanations.
        
        Args:
            question_results: List of question results from evaluation
            timestamp: Timestamp string for output file naming
            
        Returns:
            Dictionary with comprehensive multi-metric comparison results
        """
        # Extract data for multi-metric comparison
        questions = []
        correct_answers = []
        expert_explanations = []
        baseline_explanations = []
        adaptive_explanations = []
        metadata = []
        
        for qr in question_results:
            if "error" in qr:
                continue
            
            questions.append(qr["question"])
            correct_answers.append(qr["correct_answer"])  
            expert_explanations.append(qr.get("expert_explanation", ""))
            baseline_explanations.append(qr["baseline"]["teacher_explanation"])
            adaptive_explanations.append(qr["adaptive"]["teacher_explanation"])
            
            metadata.append({
                "question_id": qr["question_id"],
                "baseline_score": qr["baseline"]["overall_score"],
                "adaptive_score": qr["adaptive"]["overall_score"],
                "baseline_correct": qr["baseline"]["overall_score"] == 1.0,
                "adaptive_correct": qr["adaptive"]["overall_score"] == 1.0,
                "correct_answer": qr["correct_answer"],
                "baseline_predicted": qr["baseline"]["predicted"],
                "adaptive_predicted": qr["adaptive"]["predicted"]
            })
        
        if len(questions) == 0:
            return {
                "error": "No valid question results found",
                "summary": {}
            }
        
        print(f"\nRunning multi-metric judge on {len(questions)} explanations...")
        print(f"Comparing: adaptive vs baseline")
        print(f"Metrics: Conceptual Accuracy, Pedagogical Clarity, Misconception Avoidance,")
        print(f"         Completeness, Accessibility, Engagement Potential\n")
        
        # Run batch multi-metric comparison
        judge_results = batch_multi_metric_comparison(
            questions=questions,
            correct_answers=correct_answers, 
            explanations_a=adaptive_explanations,
            explanations_b=baseline_explanations,
            label_a="adaptive",
            label_b="baseline",
            expert_explanations=expert_explanations
        )
        
        # Add metadata to individual results
        for i, meta in enumerate(metadata):
            if i < len(judge_results["individual_results"]):
                judge_results["individual_results"][i]["metadata"] = meta
        
        # Print summary
        summary = judge_results["summary"]
        print(f"{'='*80}")
        print("MULTI-METRIC JUDGE RESULTS")
        print(f"{'='*80}")
        print(f"Total Comparisons: {summary['total_comparisons']}")
        print(f"Successful Comparisons: {summary['successful_comparisons']}")
        
        print(f"\nOverall Winners:")
        print(f"  Adaptive: {summary['overall_winners']['adaptive_wins']} ({summary['overall_winners']['adaptive_win_rate']:.1%})")
        print(f"  Baseline: {summary['overall_winners']['baseline_wins']} ({summary['overall_winners']['baseline_win_rate']:.1%})")
        print(f"  Ties:     {summary['overall_winners']['ties']}")
        
        print(f"\nScore Aggregates (0-10 scale per metric):")
        print(f"  Adaptive: {summary['score_aggregates']['adaptive']['average_per_comparison']:.2f}/60 avg per question")
        print(f"            {summary['score_aggregates']['adaptive']['average_per_metric']:.2f}/10 avg per metric")
        print(f"  Baseline: {summary['score_aggregates']['baseline']['average_per_comparison']:.2f}/60 avg per question")
        print(f"            {summary['score_aggregates']['baseline']['average_per_metric']:.2f}/10 avg per metric")
        
        print(f"\nMetric Breakdown (average scores 0-10):")
        metrics = ["conceptual_accuracy", "pedagogical_clarity", "misconception_avoidance", 
                   "completeness", "accessibility", "engagement_potential"]
        
        for metric in metrics:
            adaptive_avg = summary['metric_breakdown']['adaptive_averages'][metric]
            baseline_avg = summary['metric_breakdown']['baseline_averages'][metric]
            adaptive_wins = summary['metric_breakdown']['adaptive_metric_wins'][metric]
            baseline_wins = summary['metric_breakdown']['baseline_metric_wins'][metric]
            
            print(f"  {metric.replace('_', ' ').title()}:")
            print(f"    Adaptive: {adaptive_avg:.2f}/10 (wins: {adaptive_wins})")
            print(f"    Baseline: {baseline_avg:.2f}/10 (wins: {baseline_wins})")
        
        # Save results
        judge_output = self.results_dir / f"judge_eval_{timestamp}.json"
        judge_output.write_text(
            json.dumps(judge_results, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"\n{'='*80}")
        print(f"Results saved to: {judge_output}")
        print(f"{'='*80}\n")
        
        return judge_results