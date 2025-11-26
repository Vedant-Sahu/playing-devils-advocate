"""Minimal script to run the DSPy teacher-student pipeline on a single GPQA question.

This script demonstrates how to:
  1. Load (or point to) the compiled DSPy program produced by compile_dspy_program.py
  2. Pull a question from either the standard GPQA cache or a curated JSON file
  3. Format the question into the multiple-choice structure expected by the DSPy program
  4. Configure the DSPy runtime (LLM, temperature, cache)
  5. Execute the refined teacher → student → final-answer pipeline and display the outputs

Usage example (default GPQA cache):
    python scripts/run_dspy_example.py --compiled results/dspy_compiled.pkl --question-index 0

Usage example (custom curated dataset):
    python scripts/run_dspy_example.py \
        --compiled results/dspy_compiled.pkl \
        --custom-cache-file data/cache/my_curated_physics.json \
        --question-index 4

After the run, inspect the printed explanations, critiques, and final answer.
You can copy the `final_answer_letter` into your evaluation harness or compare
`initial_explanation` vs `refined_explanation` to see how DSPy changed the prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dspy_pipeline import (
    DSPyRuntimeConfig,
    configure_dspy_runtime,
    get_compiled_program,
    run_physics_program,
)
from src.utils.gpqa_loader import GPQALoader
from src.utils.gpqa_sampler import format_quiz_question


def _load_question(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and format a GPQA question into DSPy-ready fields."""

    loader = GPQALoader(
        subset=args.subset,
        domain=args.domain,
        custom_cache_file=args.custom_cache_file,
    )

    if args.question_index < 0 or args.question_index >= len(loader):
        raise IndexError(
            f"question_index {args.question_index} is out of range for dataset of size {len(loader)}"
        )

    raw_question = loader[args.question_index]
    formatted = format_quiz_question(raw_question, seed=args.option_seed)
    return {
        "id": formatted["id"],
        "question_text": formatted["question"],
        "options": formatted["options"],
        "correct_letter": formatted["correct"],
        "raw": raw_question,
    }


def _configure_runtime(args: argparse.Namespace) -> None:
    """Configure the DSPy runtime with the desired model/temperature/cache settings."""

    config = DSPyRuntimeConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        teacher_persona=args.teacher_persona,
        custom_cache_file=args.custom_cache_file,
    )
    configure_dspy_runtime(config)


def _display_results(question: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Pretty-print the key artifacts for quick inspection or downstream logging."""

    payload = {
        "question_id": question["id"],
        "question_text": question["question_text"],
        "options": question["options"],
        "initial_explanation": result.get("initial_explanation"),
        "refined_explanation": result.get("refined_explanation"),
        "student_critiques": result.get("critiques", []),
        "teacher_metadata": result.get("teacher_metadata", {}),
        "final_answer_letter": result.get("final_answer_letter"),
        "reasoning_trace": result.get("reasoning_trace"),
        "final_answer_summary": result.get("final_answer_summary"),
    }

    print("\n=== DSPy Run Artifacts ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        "\nNext steps: copy the final answer letter into your grading harness or compare\n"
        "initial vs refined explanations to assess how DSPy adjusted the prompt."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an optimized DSPy program on one question")

    # Question/data controls
    parser.add_argument("--subset", default="gpqa_main", help="GPQA subset (ignored if custom file)")
    parser.add_argument("--domain", default="Physics", help="Domain filter for GPQA dataset")
    parser.add_argument("--custom-cache-file", default=None, help="Path to curated GPQA-style JSON")
    parser.add_argument(
        "--question-index",
        type=int,
        default=0,
        help="Which question index to run from the chosen dataset",
    )
    parser.add_argument(
        "--option-seed",
        type=int,
        default=123,
        help="Seed used when shuffling multiple-choice options",
    )

    # DSPy runtime + compiled program controls
    parser.add_argument("--compiled", required=True, help="Path to compiled DSPy program (.pkl)")
    parser.add_argument("--model-name", default="gpt-4o-mini", help="LLM for inference")
    parser.add_argument("--temperature", type=float, default=0.0, help="Inference temperature")
    parser.add_argument("--disable-cache", action="store_true", help="Disable DSPy cache store")
    parser.add_argument("--cache-path", default=None, help="Directory for DSPy cache files")
    parser.add_argument("--teacher-persona", default="general", help="Persona input to DSPy program")

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    question = _load_question(args)
    _configure_runtime(args)

    program = get_compiled_program(args.compiled)
    result = run_physics_program(
        program,
        question_text=question["question_text"],
        options=question["options"],
        teacher_persona=args.teacher_persona,
    )

    _display_results(question, result)


if __name__ == "__main__":
    main()
