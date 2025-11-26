"""DSPy program definition for the teacher-student refinement loop."""

from __future__ import annotations

from typing import Dict, List, Optional

import dspy

from src.config import PERSONAS

from .signatures import FinalAnswerSignature, StudentCritiqueSignature, TeacherSignature
from .base_prompts import (
    TEACHER_BASE_PROMPT,
    STUDENT_CRITIC_BASE_PROMPT,
    FINAL_ANSWER_BASE_PROMPT,
)


def _format_feedback_block(critiques: List[Dict[str, str]]) -> str:
    """Turn structured critiques into a compact prompt for the teacher."""

    if not critiques:
        return "No student issues were raised."

    lines: List[str] = ["TOP STUDENT FEEDBACK:"]
    for idx, fb in enumerate(critiques, start=1):
        persona = fb.get("persona", "student")
        issue = fb.get("primary_issue", "(missing issue)")
        quote = fb.get("supporting_quote", "")
        fix = fb.get("fix_suggestion", "")
        sev = fb.get("severity_rationale", "")
        conf = fb.get("confidence", "")
        lines.append(
            "\n".join(
                [
                    f"{idx}. Persona: {persona}",
                    f"   Issue: {issue}",
                    f"   Quote: {quote}",
                    f"   Fix: {fix}",
                    f"   Why it matters: {sev}",
                    f"   Confidence: {conf}",
                ]
            )
        )
    return "\n".join(lines)


class PhysicsProgram(dspy.Module):
    """Multi-role DSPy program for GPQA physics questions."""

    def __init__(
        self,
        personas: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.personas = personas or PERSONAS
        if not self.personas:
            raise ValueError("At least one persona is required for student critiques.")

        # Seed DSPy modules with rich, task-specific instructions so that
        # GEPA optimizes these detailed prompts rather than minimal docstrings.
        self.teacher = dspy.Predict(TeacherSignature)
        self.teacher.signature.instructions = TEACHER_BASE_PROMPT

        self.final_answerer = dspy.Predict(FinalAnswerSignature)
        self.final_answerer.signature.instructions = FINAL_ANSWER_BASE_PROMPT

        self.student_modules: Dict[str, dspy.Module] = {
            persona: dspy.Predict(StudentCritiqueSignature)
            for persona in self.personas
        }
        for module in self.student_modules.values():
            module.signature.instructions = STUDENT_CRITIC_BASE_PROMPT

    def forward(
        self,
        question_text: str,
        options: List[str],
        teacher_persona: str = "general",
    ) -> Dict[str, str]:
        """Run teacher → student critiques → refined teacher → final answer."""

        # Initial teacher pass (iteration 0)
        teacher_round_0 = self.teacher(
            question_text=question_text,
            persona=teacher_persona,
            iteration_index=0,
            student_feedback="",
        )

        # Collect critiques from each persona
        critiques: List[Dict[str, str]] = []
        for persona in self.personas:
            student = self.student_modules[persona]
            critique = student(
                explanation=teacher_round_0.core_explanation,
                question_text=question_text,
                persona=persona,
                iteration_index=0,
            )
            critique_payload = {
                "persona": persona,
                "primary_issue": getattr(critique, "primary_issue", ""),
                "supporting_quote": getattr(critique, "supporting_quote", ""),
                "fix_suggestion": getattr(critique, "fix_suggestion", ""),
                "severity_rationale": getattr(critique, "severity_rationale", ""),
                "confidence": getattr(critique, "confidence", ""),
            }
            critiques.append(critique_payload)

        feedback_block = _format_feedback_block(critiques)

        # Refined teacher pass (iteration 1)
        teacher_round_1 = self.teacher(
            question_text=question_text,
            persona=teacher_persona,
            iteration_index=1,
            student_feedback=feedback_block,
        )

        refined_explanation = teacher_round_1.core_explanation

        # Final answer selection using refined explanation
        final_answer = self.final_answerer(
            refined_explanation=refined_explanation,
            question_text=question_text,
            options=options,
        )

        return {
            "initial_explanation": getattr(teacher_round_0, "core_explanation", ""),
            "refined_explanation": refined_explanation,
            "critiques": critiques,
            "teacher_metadata": {
                "misconceptions_addressed": getattr(
                    teacher_round_1, "misconceptions_addressed", ""
                ),
                "difficult_detail_walkthrough": getattr(
                    teacher_round_1, "difficult_detail_walkthrough", ""
                ),
                "follow_up_focus": getattr(teacher_round_1, "follow_up_focus", ""),
            },
            "final_answer_letter": getattr(final_answer, "final_answer_letter", None),
            "reasoning_trace": getattr(final_answer, "reasoning_trace", ""),
            "final_answer_summary": getattr(final_answer, "final_answer_summary", ""),
        }
