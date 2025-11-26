"""CLI to compile the DSPy physics program with GEPA only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import dspy

# Latest DSPy exposes optimizers at the top-level module, matching docs like
# https://dspy.ai/tutorials/gepa_aime/ which demonstrate `from dspy import GEPA`.
from dspy import GEPA


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Make sure the local src/ package root is importable when running this script
# directly (e.g., `python scripts/compile_dspy_program.py`). This lets us
# import the in-repo `dspy_pipeline` package without requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dspy_pipeline import (
    DSPyRuntimeConfig,
    PhysicsProgram,
    configure_dspy_runtime,
    multiple_choice_accuracy,
    gepa_multiple_choice_metric,
    run_physics_program,
    sample_gpqa_dataset,
    save_compiled_program,
)


def _evaluate_program(program: PhysicsProgram, devset: Sequence[dspy.Example]) -> float:
    total = 0.0
    iterator = (
        tqdm(devset, desc="Evaluating dev set", leave=False)
        if tqdm is not None
        else devset
    )
    for example in iterator:
        prediction = run_physics_program(
            program,
            question_text=example.question_text,
            options=example.options,
            teacher_persona="general",
        )
        total += multiple_choice_accuracy(example, prediction)
    return total / max(1, len(devset))


def compile_program(args: argparse.Namespace) -> None:
    config = DSPyRuntimeConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        teacher_persona=args.teacher_persona,
        custom_cache_file=args.custom_cache_file,
    )
    configure_dspy_runtime(config)

    trainset, devset = sample_gpqa_dataset(
        subset=args.subset,
        domain=args.domain,
        num_train=args.num_train,
        num_dev=args.num_dev,
        seed=args.seed,
        custom_cache_file=args.custom_cache_file,
    )

    program = PhysicsProgram()

    metric = multiple_choice_accuracy

    # GEPA-only optimization, following the latest tutorial style:
    #
    #   from dspy import GEPA
    #   optimizer = GEPA(metric=metric_with_feedback, auto="light", ...)
    #   optimized_program = optimizer.compile(program, trainset=train_set, valset=val_set)
    #
    print(f"Running GEPA optimization on {len(trainset)} train examples...")

    # GEPA requires a separate "reflection" LM it can use to introspect on the
    # program's behavior and propose new instructions. For now we reuse the same
    # model specified via --model-name as the reflection LM, mirroring the
    # tutorial pattern `GEPA(..., reflection_lm=dspy.LM(...))`.
    reflection_lm = dspy.LM(args.model_name, temperature=args.temperature)

    optimizer = GEPA(
        metric=gepa_multiple_choice_metric,
        # We disable the automatic budget presets (`auto=None`) and instead
        # set an explicit cap on metric calls so the search stays cheap.
        auto="light",
        track_stats=True,
        # Tighten GEPA's search budget so we don't explode the number of
        # rollouts/metric calls on small datasets. You can bump this up later
        # if you want a more exhaustive search.
        #max_metric_calls=200,
        reflection_minibatch_size=2,
        reflection_lm=reflection_lm,
    )
    compiled: PhysicsProgram = optimizer.compile(
        program,
        trainset=trainset,
        valset=devset,
    )

    print(f"Evaluating on {len(devset)} held-out dev examples...")
    dev_score = _evaluate_program(compiled, devset)
    print(f"Dev accuracy: {dev_score:.3f}")

    output_path = save_compiled_program(compiled, args.output)
    print(f"Saved compiled program to {output_path}")

    metadata = {
        "subset": args.subset,
        "domain": args.domain,
        "num_train": args.num_train,
        "num_dev": args.num_dev,
        "seed": args.seed,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "teacher_persona": args.teacher_persona,
        "custom_cache_file": args.custom_cache_file,
        "dev_accuracy": dev_score,
        "compiled_path": str(output_path),
    }
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile DSPy prompts for GPQA physics")
    parser.add_argument("--subset", default="gpqa_main", help="GPQA subset name")
    parser.add_argument("--domain", default="Physics", help="GPQA domain filter")
    parser.add_argument("--custom-cache-file", default=None, help="Path to curated GPQA-style JSON")
    parser.add_argument("--num-train", type=int, default=32, help="Training examples")
    parser.add_argument("--num-dev", type=int, default=16, help="Dev examples")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--model-name", default="gpt-4o-mini", help="LLM name for DSPy")
    parser.add_argument("--temperature", type=float, default=0.0, help="Compilation temperature")
    parser.add_argument("--teacher-persona", default="general", help="Teacher persona input")
    parser.add_argument("--output", default="results/dspy_compiled.pkl", help="Path to save compiled program")
    # GEPA has many advanced knobs; for now we keep the CLI minimal and
    # rely on the default settings (auto="light").
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    compile_program(args)


if __name__ == "__main__":
    main()
