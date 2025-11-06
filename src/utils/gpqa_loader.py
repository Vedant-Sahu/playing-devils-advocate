"""
GPQA dataset loader for physics questions.

Loads cached GPQA physics questions from local JSON files.
Questions must be downloaded first using scripts/download_gpqa.py
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import random


class GPQALoader:
    """Load GPQA physics questions from cache"""
    
    def __init__(self, subset: str = "gpqa_main", domain: str = "Physics"):
        """
        Initialize GPQA loader.
        
        Args:
            subset: One of 'gpqa_main', 'gpqa_extended', or 'gpqa_diamond'
            domain: Domain filter (default: 'Physics')
        
        Raises:
            FileNotFoundError: If cache file doesn't exist
            ValueError: If subset is invalid
        """
        valid_subsets = ["gpqa_main", "gpqa_extended", "gpqa_diamond"]
        if subset not in valid_subsets:
            raise ValueError(f"Invalid subset. Must be one of {valid_subsets}")
        
        self.subset = subset
        self.domain = domain
        self.questions = self._load_from_cache()
    
    def _load_from_cache(self) -> List[Dict]:
        """Load questions from cached JSON file"""
        base_dir = Path(__file__).resolve().parents[2]
        domain_key = "Physics" if str(self.domain).lower() == "physics" else str(self.domain)
        cache_file = base_dir / "data" / "cache" / f"{self.subset}_{domain_key}_train.json"
        
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}\n"
                f"Run 'python scripts/download_gpqa.py' to download the dataset."
            )
        
        print(f"Loading GPQA cache from: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if not questions:
            raise ValueError(f"No questions found in {cache_file}")
        
        print(f"Loaded {len(questions)} {self.domain} questions from {self.subset}")
        return questions
    
    def get_question(self, index: int) -> Dict:
        """
        Get question by index.
        
        Args:
            index: Question index (0 to len-1)
        
        Returns:
            Question dict with keys: id, question, correct_answer, 
            incorrect_answers, expert_explanation, domain
        """
        if not 0 <= index < len(self.questions):
            raise IndexError(f"Index {index} out of range [0, {len(self.questions)})")
        return self.questions[index]
    
    def get_random_question(self) -> Dict:
        """Get random question from dataset"""
        return random.choice(self.questions)
    
    def get_batch(self, start: int, size: int) -> List[Dict]:
        """
        Get batch of questions.
        
        Args:
            start: Starting index
            size: Number of questions to return
        
        Returns:
            List of question dicts
        """
        end = min(start + size, len(self.questions))
        return self.questions[start:end]
    
    def __len__(self) -> int:
        """Return number of questions"""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict:
        """Allow indexing: loader[0]"""
        return self.get_question(idx)
    
    def __iter__(self):
        """Allow iteration: for q in loader"""
        return iter(self.questions)


if __name__ == "__main__":
    # Test the loader
    try:
        loader = GPQALoader("gpqa_main", "Physics")
        print(f"\nTotal questions: {len(loader)}")
        print("\nFirst question:")
        q = loader[0]
        print(f"ID: {q['id']}")
        print(f"Question: {q['question'][:100]}...")
        print(f"Correct answer: {q['correct_answer']}")
        print(f"Domain: {q['domain']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")