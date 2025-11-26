"""DSPy signatures for the physics teacher-student workflow."""

from __future__ import annotations

import dspy


class TeacherSignature(dspy.Signature):
    """Explain the physics question with a focus on misconceptions and tricky details."""

    question_text = dspy.InputField(desc="Physics prompt the teacher must explain")
    persona = dspy.InputField(
        desc="Target student persona (advanced, struggling, etc.)", default="general"
    )
    iteration_index = dspy.InputField(
        desc="0 for the first pass, >0 for refinement iterations", default=0
    )
    student_feedback = dspy.InputField(
        desc="Structured critiques from students; empty string on the first pass",
        default="",
    )
    core_explanation = dspy.OutputField(
        desc="Step-by-step reasoning that walks through the toughest parts of the problem"
    )
    misconceptions_addressed = dspy.OutputField(
        desc="List of misconceptions or likely confusions explicitly corrected"
    )
    difficult_detail_walkthrough = dspy.OutputField(
        desc="Detailed treatment of the hardest quantitative or conceptual step"
    )
    follow_up_focus = dspy.OutputField(
        desc="What should the next iteration or student focus on if confusion remains"
    )


class StudentCritiqueSignature(dspy.Signature):
    """Produce a high-quality student critique targeting the teacher explanation."""

    explanation = dspy.InputField(desc="Most recent teacher explanation")
    question_text = dspy.InputField(desc="Original GPQA question for grounding")
    persona = dspy.InputField(desc="Student persona issuing the critique")
    iteration_index = dspy.InputField(desc="Refinement round index", default=0)
    primary_issue = dspy.OutputField(desc="Single most important flaw or missing detail")
    supporting_quote = dspy.OutputField(desc="Direct quote that shows the issue")
    fix_suggestion = dspy.OutputField(desc="Concrete guidance for how to repair the issue")
    severity_rationale = dspy.OutputField(desc="Why this matters for student understanding")
    confidence = dspy.OutputField(desc="Confidence 0-1 or 'low/med/high'")


class FinalAnswerSignature(dspy.Signature):
    """Select the best multiple-choice answer and justify it using the refined explanation."""

    refined_explanation = dspy.InputField(desc="Latest teacher explanation after critiques")
    question_text = dspy.InputField(desc="Original multiple-choice question text")
    options = dspy.InputField(desc="List of answer options like 'A) ...'")
    final_answer_letter = dspy.OutputField(desc="One of A/B/C/D only")
    reasoning_trace = dspy.OutputField(
        desc="Short derivation referencing steps/lines in the refined explanation"
    )
    final_answer_summary = dspy.OutputField(
        desc="One-sentence summary with units or qualitative conclusion"
    )
