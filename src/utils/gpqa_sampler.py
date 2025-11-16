"""
GPQA Question Sampler - Utilities for sampling and formatting GPQA questions.

This module provides functions to deterministically sample GPQA questions and
format them into the quiz structure expected by the grading agent.
"""

import random
from typing import Dict, List, Any, Tuple

from src.utils.gpqa_loader import GPQALoader


def format_quiz_question(
    gpqa_entry: Dict[str, Any],
    seed: int = 17
) -> Dict[str, Any]:
    """
    Format a GPQA question into the quiz structure expected by the grading agent.
    
    This function:
    - Extracts the question, correct answer, and incorrect answers
    - Shuffles options deterministically based on seed
    - Assigns letter labels (A, B, C, D)
    - Tracks which letter corresponds to the correct answer
    
    Args:
        gpqa_entry: Raw GPQA question dict from GPQALoader
        seed: Random seed for deterministic option shuffling
        
    Returns:
        Dict with keys: id, question, options (list of answers), correct (label), expert_explanation
    """
    rng = random.Random(seed)
    
    # Extract question metadata
    record_id = gpqa_entry.get("record_id", gpqa_entry.get("id"))
    question = gpqa_entry.get("question", "").strip()
    explanation = gpqa_entry.get("expert_explanation", "").strip()
    
    # Build option items with correct/incorrect labels
    option_labels = ["A", "B", "C", "D"]
    option_items = [("correct", gpqa_entry.get("correct_answer", "").strip())] 
    for wrong_answer in (gpqa_entry.get("incorrect_answers") or [])[:3]:
        option_items.append(("incorrect", wrong_answer.strip()))
    
    # Deterministic shuffle
    rng.shuffle(option_items)
    
    # Format options with option labels
    options_text = [f"{option_labels[i]}) {option_items[i][1]}" for i in range(4)]
    
    # Find which letter is correct
    correct_idx = [i for i, (label, _) in enumerate(option_items) if label == "correct"][0]
    correct_label = option_labels[correct_idx]
    
    return {
        "id": str(record_id),
        "question": question,
        "options": options_text,
        "correct": correct_label,
        "expert_explanation": explanation
    }


def create_gpqa_quiz(
    subset: str = "gpqa_main",
    domain: str = "Physics",
    seed: int = 17,
    index: int = None,
    num_questions: int = 1
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Create a formatted quiz from GPQA questions.
    
    Args:
        subset: GPQA subset to sample from (e.g., "gpqa_main")
        domain: Domain to sample from (e.g., "Physics", "Chemistry", "Biology")
        seed: Random seed for deterministic sampling and shuffling
        index: Specific starting index
        num_questions: Number of questions to include in quiz
        
    Returns:
        Tuple of (quiz_list, sampled_indices)
    """
    rng = random.Random(seed)
    loader = GPQALoader(subset, domain)
    
    # Sample indices
    if index is not None:
        if index + num_questions > len(loader):
            raise ValueError(
                f"Cannot sample {num_questions} questions starting at index {index}."
                f"Dataset has {len(loader)} questions."
            )
        indices = list(range(index, index + num_questions))
    else:
        indices = rng.sample(range(len(loader)), num_questions)
    
    # Format questions
    quiz = []
    for idx in indices:
        gpqa_entry = loader[idx]
        formatted = format_quiz_question(gpqa_entry, seed=seed+idx)  # Vary seed per question
        quiz.append(formatted)
    
    return quiz, indices