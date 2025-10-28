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
from pathlib import Path
import argparse

def download_and_cache_gpqa(subset="gpqa_main", domain="Physics"):
    """
    Download GPQA and save physics questions to cache.
    
    Args:
        subset: One of 'gpqa_main', 'gpqa_extended', or 'gpqa_diamond'
        domain: Domain to filter (default: 'Physics')
    
    Returns:
        Number of questions downloaded
    """
    print(f"Downloading GPQA {subset} from Hugging Face...")
    print(f"Source: https://huggingface.co/datasets/Idavidrein/gpqa")
    
    dataset = load_dataset("Idavidrein/gpqa", subset)

    # Filter for physics questions
    physics_questions = []
    for example in dataset['train']:
        domain_field = example.get('High-level domain')
        if domain in domain_field:
            physics_questions.append({
                'id': len(physics_questions),
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
    
    # Save to cache
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = cache_dir / f"{subset}_{domain}.json"
    with open(output_file, 'w') as f:
        json.dump(physics_questions, f, indent=2)
    
    print(f"✓ Saved {len(physics_questions)} physics questions to {output_file}")
    return len(physics_questions)


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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPQA Dataset Download Script")
    print("=" * 60)
    
    total = download_and_cache_gpqa(args.subset, args.domain)
    
    print("\n" + "=" * 60)
    print(f"Download complete! {total} questions saved.")
    print("\nNext steps:")
    print("1. Questions are cached in data/cache/")
    print("2. Use GPQALoader to load them in your code")
    print("=" * 60)