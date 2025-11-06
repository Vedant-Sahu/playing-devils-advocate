"""
Download GPQA dataset and cache physics questions locally.

This script downloads the GPQA benchmark from Hugging Face and extracts
only the physics questions for use in our experiments.

Prerequisites:
1. Request access: https://huggingface.co/datasets/Idavidrein/gpqa
2. Get HF token: https://huggingface.co/settings/tokens 
3. Login: python -c "from huggingface_hub import login; login()"
   OR set HF_TOKEN in .env file

Usage:
    python scripts/download_gpqa.py

Note: This will download 448 questions from gpqa_main subset.
"""

from datasets import load_dataset
import json
import random
from pathlib import Path
import argparse

def download_and_cache_gpqa(
    subset: str = "gpqa_main",
    domain: str = "Physics",
    example_size: int = 5,
    seed: int = 17
):
    """
    Download GPQA and save physics questions to cache.
    
    Args:
        subset: One of 'gpqa_main', 'gpqa_extended', or 'gpqa_diamond'
        domain: Domain to filter (default: 'Physics')
        example_size: Number of questions to reserve for examples
        seed: Random seed for reproducible splitting
    
    Returns:
        Dict with counts: {"example": int, "train": int, "total": int}
    """
    print(f"Downloading GPQA {subset} from Hugging Face...")
    print(f"Source: https://huggingface.co/datasets/Idavidrein/gpqa")
    
    dataset = load_dataset("Idavidrein/gpqa", subset)

    # Filter for domain questions
    domain_questions = []
    for example in dataset['train']:
        domain_field = example.get('High-level domain')
        if domain in domain_field:
            domain_questions.append({
                'id': len(domain_questions),
                'record_id': example.get('Record ID'),
                'question': example['Question'],
                'correct_answer': example['Correct Answer'],
                'incorrect_answers': [
                    example['Incorrect Answer 1'],
                    example['Incorrect Answer 2'],
                    example['Incorrect Answer 3']
                ],
                'expert_explanation': example.get('Explanation', ''),
                'domain': domain_field,
                'subdomain': example.get('Subdomain', ''),
                'difficulty': example.get("Writer's Difficulty Estimate", ''),
                'non_expert_accuracy': example['Non-Expert Validator Accuracy'],
                'expert_accuracy': example['Expert Validator Accuracy']
            })

    total_questions = len(domain_questions)
    print(f"Found {total_questions} {domain} questions")
    
    if total_questions < example_size + 1:
        raise ValueError(
            f"Not enough questions ({total_questions}) to split. "
            f"Need at least {example_size + 1} questions."
        )
    
    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(domain_questions)

    # Split into example/train
    example_questions = domain_questions[:example_size]
    train_questions = domain_questions[example_size:]

    # Reassign IDs for each split
    for i, q in enumerate(example_questions):
        q['id'] = i
    for i, q in enumerate(train_questions):
        q['id'] = i
    
    # Save to cache
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'example': example_questions,
        'train': train_questions
    }

    counts = {}
    for split_name, questions in splits.items():
        output_file = cache_dir / f"{subset}_{domain}_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(questions, f, indent=2)
        counts[split_name] = len(questions)
        print(f"Saved {len(questions)} questions to {output_file}")
    
    # Also save metadata about the split
    metadata = {
        'subset': subset,
        'domain': domain,
        'seed': seed,
        'example_size': example_size,
        'counts': counts,
        'total': total_questions
    }
    
    metadata_file = cache_dir / f"{subset}_{domain}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    counts['total'] = total_questions
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GPQA physics questions"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="gpqa_main",
        choices=["gpqa_main", "gpqa_extended", "gpqa_diamond"],
        help="GPQA subset to download (default: gpqa_main)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Physics",
        help="Domain to filter (default: Physics)"
    )
    parser.add_argument(
        "--example_size",
        type=int,
        default=5,
        help="Number of questions to reserve for examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducible splitting"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPQA Dataset Download Script")
    print("=" * 60)
    
    counts = download_and_cache_gpqa(args.subset, args.domain, args.example_size, args.seed)
    
    print("\n" + "=" * 60)
    print(f"Download complete! {counts['total']} questions saved.")
    print("Dataset Split Summary:")
    print(f"  Example: {counts['example']} questions")
    print(f"  Train:   {counts['train']} questions")
    print("\nNext steps:")
    print("1. Questions are cached in data/cache/")
    print("2. Use GPQALoader to load them in your code")
    print("=" * 60)