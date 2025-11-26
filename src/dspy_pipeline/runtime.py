"""Runtime helpers for configuring and executing DSPy programs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from .program import PhysicsProgram


@dataclass
class DSPyRuntimeConfig:
    """Configuration for DSPy language model runtime.

    This mirrors the current DSPy docs pattern:

        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
    """

    model_name: str = os.getenv("DSPY_MODEL_NAME", "openai/gpt-4o-mini")
    temperature: float = float(os.getenv("DSPY_TEMPERATURE", "0.0"))
    teacher_persona: str = "general"
    custom_cache_file: Optional[str] = None


def configure_dspy_runtime(config: DSPyRuntimeConfig) -> None:
    """Configure DSPy with the chosen LLM.

    Uses dspy.LM + dspy.configure, which is the recommended, minimal
    configuration pattern in the latest DSPy tutorials.
    """

    # Ensure OpenAI key is present for openai/* models; dspy.LM will
    # read it from the environment.
    if config.model_name.startswith("openai/") and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set to use OpenAI-backed DSPy models.")

    lm = dspy.LM(config.model_name, temperature=config.temperature)
    dspy.configure(lm=lm)


def run_physics_program(
    program: PhysicsProgram,
    question_text: str,
    options: List[str],
    teacher_persona: str = "general",
) -> Dict[str, Any]:
    """Execute the physics DSPy program for a single multiple-choice question."""

    return program(
        question_text=question_text,
        options=options,
        teacher_persona=teacher_persona,
    )


def save_compiled_program(program: PhysicsProgram, path: str | Path) -> Path:
    """Persist a compiled DSPy program to disk using DSPy's save API.

    We use `program.save(..., save_program=True)` as recommended in the DSPy
    "Saving and Loading" tutorial. The `path` is treated as a *directory* that
    will hold the serialized program and its metadata.
    """

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    # DSPy handles serialization (via cloudpickle under the hood) and stores
    # any required metadata alongside the program state.
    program.save(str(target), save_program=True)
    return target


def get_compiled_program(path: str | Path) -> PhysicsProgram:
    """Load a compiled DSPy program from disk via dspy.load()."""

    program = dspy.load(str(path))
    if not isinstance(program, PhysicsProgram):
        raise TypeError("Loaded object is not a PhysicsProgram.")
    return program
