"""Dataset helpers for DSPy training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import random

import dspy

from src.utils.gpqa_loader import GPQALoader
from src.utils.gpqa_sampler import format_quiz_question


@dataclass
class GPQAExample:
    question_text: str
    options: List[str]
    correct_letter: str
    record_id: str

    def to_dspy_example(self) -> dspy.Example:
        return dspy.Example(
            question_text=self.question_text,
            options=self.options,
            correct_letter=self.correct_letter,
            record_id=self.record_id,
        ).with_inputs("question_text", "options")


def _format_gpqa_entry(entry: Dict[str, Any], option_seed: int) -> GPQAExample:
    formatted = format_quiz_question(entry, seed=option_seed)
    return GPQAExample(
        question_text=formatted["question"],
        options=formatted["options"],
        correct_letter=str(formatted["correct"]).strip().upper(),
        record_id=str(formatted["id"]),
    )


def sample_gpqa_dataset(
    subset: str = "gpqa_main",
    domain: str = "Physics",
    num_train: int = 32,
    num_dev: int = 16,
    seed: int = 17,
    custom_cache_file: str | None = None,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Create train/dev splits of GPQA questions for DSPy optimization."""

    loader = GPQALoader(subset=subset, domain=domain, custom_cache_file=custom_cache_file)
    total = len(loader)
    if num_train + num_dev > total:
        raise ValueError(
            f"Requested {num_train + num_dev} questions but dataset only has {total}."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(total), num_train + num_dev)
    train_ids = indices[:num_train]
    dev_ids = indices[num_train : num_train + num_dev]

    train_examples = [
        _format_gpqa_entry(loader[idx], option_seed=seed + idx).to_dspy_example()
        for idx in train_ids
    ]
    dev_examples = [
        _format_gpqa_entry(loader[idx], option_seed=seed + idx + 999).to_dspy_example()
        for idx in dev_ids
    ]

    return train_examples, dev_examples
