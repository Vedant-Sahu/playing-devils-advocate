"""Helpers to integrate DSPy-compiled programs into the LangGraph runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runtime import (
    DSPyRuntimeConfig,
    configure_dspy_runtime,
    get_compiled_program,
    run_physics_program,
)
from .program import PhysicsProgram


_BACKEND_ENV = "TEACHER_PROMPT_BACKEND"
_DSPY_FLAG = "dspy"
_PROGRAM_CACHE: Optional[PhysicsProgram] = None
_PROGRAM_PATH: Optional[str] = None
_RUNTIME_READY = False


def should_use_dspy_teacher_backend() -> bool:
    """Return True if the environment requests the DSPy teacher backend."""

    value = os.getenv(_BACKEND_ENV, "llm").strip().lower()
    return value == _DSPY_FLAG


def _ensure_runtime() -> None:
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return

    model_name = os.getenv("DSPY_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4o-mini")
    compile_temp = float(os.getenv("DSPY_INFER_TEMPERATURE", os.getenv("DSPY_TEMPERATURE", "0.0")))
    cache_path = os.getenv("DSPY_CACHE_PATH")
    teacher_persona = os.getenv("DSPY_TEACHER_PERSONA", "general")
    custom_cache_file = os.getenv("DSPY_CUSTOM_CACHE_FILE")

    config = DSPyRuntimeConfig(
        model_name=model_name,
        temperature=compile_temp,
        cache=True,
        cache_path=cache_path,
        teacher_persona=teacher_persona,
        custom_cache_file=custom_cache_file,
    )
    configure_dspy_runtime(config)
    _RUNTIME_READY = True


def _ensure_program_loaded() -> PhysicsProgram:
    global _PROGRAM_CACHE, _PROGRAM_PATH

    compiled_path = os.getenv("DSPY_COMPILED_PATH")
    if not compiled_path:
        raise EnvironmentError("DSPY_COMPILED_PATH must be set to use the DSPy backend.")

    compiled_path = str(Path(compiled_path).expanduser())
    if _PROGRAM_CACHE is None or _PROGRAM_PATH != compiled_path:
        _PROGRAM_CACHE = get_compiled_program(compiled_path)
        _PROGRAM_PATH = compiled_path
    return _PROGRAM_CACHE


def run_dspy_teacher_pass(
    gpqa_question: Dict[str, Any],
    teacher_persona: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the compiled DSPy program and return its outputs for integration."""

    if not should_use_dspy_teacher_backend():
        raise RuntimeError("DSPy teacher backend is not enabled; cannot run DSPy program.")

    question_text = str(gpqa_question.get("question", "")).strip()
    options = gpqa_question.get("options")

    if not question_text:
        raise ValueError("gpqa_question is missing 'question' text required for DSPy program.")
    if not isinstance(options, list) or not options:
        raise ValueError("gpqa_question must include 'options' for the DSPy program.")

    _ensure_runtime()
    program = _ensure_program_loaded()

    persona = teacher_persona or os.getenv("DSPY_TEACHER_PERSONA", "general")

    result = run_physics_program(
        program,
        question_text=question_text,
        options=options,
        teacher_persona=persona,
    )

    payload: Dict[str, Any] = {
        "backend": "dspy",
        "question_id": gpqa_question.get("id"),
        "initial_explanation": result.get("initial_explanation"),
        "refined_explanation": result.get("refined_explanation"),
        "critiques": result.get("critiques", []),
        "teacher_metadata": result.get("teacher_metadata", {}),
        "final_answer_letter": result.get("final_answer_letter"),
        "reasoning_trace": result.get("reasoning_trace"),
        "final_answer_summary": result.get("final_answer_summary"),
    }
    return payload
