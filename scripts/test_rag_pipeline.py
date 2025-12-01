#!/usr/bin/env python
"""
Test script for RAG-enabled pipeline on 10 test questions.

Usage:
    python scripts/test_rag_pipeline.py

Requirements:
    - USE_RAG=1 in .env
    - TEACHER_MODEL=o3 in .env
    - ANSWER_MODEL=o3 in .env (or MODEL_NAME as fallback)
    - Vector store built via: python scripts/build_rag_corpus.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def check_config():
    """Verify configuration before running."""
    print("=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)
    
    # Check models
    teacher_model = os.getenv("TEACHER_MODEL", os.getenv("MODEL_NAME", "not set"))
    answer_model = os.getenv("ANSWER_MODEL", os.getenv("MODEL_NAME", "not set"))
    
    print(f"TEACHER_MODEL: {teacher_model}")
    print(f"ANSWER_MODEL:  {answer_model}")
    
    # Check RAG
    use_rag = os.getenv("USE_RAG", "").lower() in ("1", "true", "yes")
    print(f"USE_RAG:       {use_rag} (env value: {os.getenv('USE_RAG', 'not set')})")
    
    if not use_rag:
        print("\n⚠️  WARNING: RAG is NOT enabled!")
        print("   Add USE_RAG=1 to your .env file to enable RAG.")
        response = input("Continue without RAG? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check vector store exists
    vectorstore_path = project_root / "data" / "physics_vectorstore"
    if vectorstore_path.exists():
        print(f"Vector store: ✓ Found at {vectorstore_path}")
    else:
        print(f"Vector store: ✗ NOT FOUND at {vectorstore_path}")
        if use_rag:
            print("   Run: python scripts/build_rag_corpus.py")
            sys.exit(1)
    
    # Test RAG retrieval
    if use_rag:
        print("\nTesting RAG retrieval...")
        try:
            from src.rag.retriever import retrieve_physics_context
            test_ctx = retrieve_physics_context("What is quantum entanglement?", k=2)
            if test_ctx:
                print(f"   ✓ RAG working - retrieved {len(test_ctx)} chars")
            else:
                print("   ⚠️ RAG returned empty context")
        except Exception as e:
            print(f"   ✗ RAG error: {e}")
    
    print("=" * 60)
    return use_rag


def load_test_questions():
    """Load the 10 test questions and convert to quiz format."""
    from src.utils.gpqa_sampler import format_quiz_question
    
    test_path = project_root / "data" / "cache" / "questions_test.json"
    
    if not test_path.exists():
        print(f"ERROR: Test questions not found at {test_path}")
        sys.exit(1)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        raw_questions = json.load(f)
    
    # Convert each raw question to quiz format with shuffled options
    questions = []
    for i, raw_q in enumerate(raw_questions):
        # Check if already in quiz format (has "options" and "correct" letter)
        if "options" in raw_q and "correct" in raw_q and len(raw_q.get("correct", "")) == 1:
            questions.append(raw_q)
        else:
            # Convert from raw GPQA format to quiz format
            # Use question index as part of seed for deterministic but varied shuffling
            quiz_q = format_quiz_question(raw_q, seed=42 + i)
            questions.append(quiz_q)
    
    print(f"\nLoaded {len(questions)} test questions from {test_path.name}")
    print(f"  Sample question correct answer: {questions[0].get('correct', '?')}")
    return questions


def run_single_question(question_data: dict, graph, use_rag: bool) -> dict:
    """Run pipeline on a single question and return results."""
    from src.graphs.adaptive_refinement_graph import create_initial_state
    
    q_id = question_data.get("id", "unknown")
    q_text = question_data.get("question", "")[:80]
    
    print(f"\n{'─' * 50}")
    print(f"Question: {q_id}")
    print(f"  {q_text}...")
    
    # Create initial state
    init_state = create_initial_state(
        gpqa_question=question_data,
        threshold=0.7,
        max_iters=3
    )
    
    # Run the graph
    start_time = time.time()
    final_state = graph.invoke(init_state, config={"recursion_limit": 30})
    elapsed = time.time() - start_time
    
    # Extract results
    quiz_results = final_state.get("quiz_results", {}) or {}
    is_correct = quiz_results.get("is_correct", False)
    similarity = quiz_results.get("explanation_similarity", 0.0)
    predicted_letter = final_state.get("single_answer", "?")
    correct_letter = question_data.get("correct", "?")
    iterations = final_state.get("iteration", 0)
    student_justification = final_state.get("single_explanation", "")
    
    # Look up actual answer text from options
    options = question_data.get("options", [])
    letter_to_text = {}
    for opt in options:
        if opt and len(opt) >= 2 and opt[1] == ')':
            letter_to_text[opt[0].upper()] = opt[3:].strip() if len(opt) > 3 else opt
    
    correct_text = letter_to_text.get(correct_letter.upper(), "")
    predicted_text = letter_to_text.get(predicted_letter.upper(), "")
    
    status = "✓" if is_correct else "✗"
    print(f"  Result: {status} Predicted={predicted_letter}, Correct={correct_letter}")
    print(f"  Iterations: {iterations}, Similarity: {similarity:.2f}, Time: {elapsed:.1f}s")
    
    # Get retrieved context if RAG was used
    retrieved_context = ""
    if use_rag:
        try:
            from src.agents.teacher_agent import get_last_retrieved_context
            retrieved_context = get_last_retrieved_context()
        except Exception:
            pass
    
    return {
        "id": q_id,
        "question": question_data.get("question", ""),
        "correct_letter": correct_letter,
        "correct_answer": correct_text,
        "predicted_letter": predicted_letter,
        "predicted_answer": predicted_text,
        "is_correct": is_correct,
        "student_justification": student_justification,
        "explanation_similarity": similarity,
        "iterations": iterations,
        "explanation": final_state.get("explanation", ""),
        "retrieved_context": retrieved_context[:3000] if retrieved_context else "",  # First 3000 chars
        "elapsed_seconds": elapsed,
        "rag_enabled": use_rag,
    }


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("RAG-ENABLED PIPELINE TEST")
    print("=" * 60)
    
    # Check configuration
    use_rag = check_config()
    
    # Load test questions
    questions = load_test_questions()
    
    # Import and create graph
    print("\nInitializing adaptive refinement graph...")
    from src.graphs.adaptive_refinement_graph import create_adaptive_refinement_graph
    graph = create_adaptive_refinement_graph()
    print("Graph ready.")
    
    # Run on all questions
    results = []
    correct_count = 0
    
    print(f"\n{'=' * 60}")
    print(f"RUNNING {len(questions)} QUESTIONS")
    print(f"RAG: {'ENABLED' if use_rag else 'DISABLED'}")
    print(f"{'=' * 60}")
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}]", end="")
        try:
            result = run_single_question(q, graph, use_rag)
            results.append(result)
            if result["is_correct"]:
                correct_count += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({
                "id": q.get("id", "unknown"),
                "error": str(e),
                "is_correct": False,
            })
    
    # Summary
    accuracy = (correct_count / len(questions)) * 100 if questions else 0
    avg_similarity = sum(r.get("explanation_similarity", 0) for r in results) / len(results) if results else 0
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Questions:  {len(questions)}")
    print(f"Correct:          {correct_count}")
    print(f"Accuracy:         {accuracy:.1f}%")
    print(f"Avg Similarity:   {avg_similarity:.2f}")
    print(f"RAG Enabled:      {use_rag}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_suffix = "_rag" if use_rag else "_norag"
    output_path = project_root / "results" / f"test_results_{timestamp}{rag_suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": timestamp,
        "config": {
            "rag_enabled": use_rag,
            "teacher_model": os.getenv("TEACHER_MODEL", "not set"),
            "answer_model": os.getenv("ANSWER_MODEL", "not set"),
            "num_questions": len(questions),
        },
        "summary": {
            "correct": correct_count,
            "total": len(questions),
            "accuracy_pct": accuracy,
            "avg_similarity": avg_similarity,
        },
        "results": results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
