"""DSPy integration utilities for the physics teacher-student system."""

from .signatures import (
    TeacherSignature,
    StudentCritiqueSignature,
    FinalAnswerSignature,
)
from .program import PhysicsProgram
from .metrics import multiple_choice_accuracy, gepa_multiple_choice_metric
from .datasets import sample_gpqa_dataset
from .runtime import (
    DSPyRuntimeConfig,
    configure_dspy_runtime,
    get_compiled_program,
    run_physics_program,
    save_compiled_program,
)

__all__ = [
    "TeacherSignature",
    "StudentCritiqueSignature",
    "FinalAnswerSignature",
    "PhysicsProgram",
    "multiple_choice_accuracy",
    "gepa_multiple_choice_metric",
    "sample_gpqa_dataset",
    "DSPyRuntimeConfig",
    "configure_dspy_runtime",
    "get_compiled_program",
    "run_physics_program",
    "save_compiled_program",
]
