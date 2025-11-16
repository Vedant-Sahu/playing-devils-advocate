"""
Parsing utilities for JSON extraction and value parsing.

This module provides helper functions for extracting structured data
from LLM outputs, handling various edge cases and formats.
"""

import json
import re
from typing import Any


def _extract_json(text: str) -> Any:
    """
    Extract JSON from model output with multiple fallback strategies.
    
    Tries in order:
    1. JSON within markdown code fences (```json ... ```)
    2. Direct JSON parsing of entire text
    3. Finding first {...} block
    4. Regex match for trailing JSON object
    
    Args:
        text: Raw text output from LLM
        
    Returns:
        Parsed JSON object (dict, list, etc.)
        
    Raises:
        ValueError: If no valid JSON can be extracted
        
    Example:
        >>> text = "Here's the data: ```json\\n{\"score\": 0.8}\\n```"
        >>> _extract_json(text)
        {'score': 0.8}
    """
    # Try markdown code fence
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    
    # Try direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try finding {...} block
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    
    # Try regex for trailing JSON
    match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    
    raise ValueError("Could not extract valid JSON from model output.")


def _extract_float(text: str) -> float:
    """
    Extract a float value from text, clamping to [0.0, 1.0].
    
    Searches for float patterns and returns the first match,
    clamped to valid range. Returns 0.5 as default if no match found.
    
    Args:
        text: Text containing a float value
        
    Returns:
        Float value clamped to [0.0, 1.0], or 0.5 if not found
        
    Example:
        >>> _extract_float("The score is 0.85")
        0.85
        >>> _extract_float("No numbers here")
        0.5
    """
    # Match patterns like: 0.85, .85, 1, 0, 1.0
    m = re.search(r"([01](?:\.\d+)?|0?\.\d+)", text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except Exception:
            pass
    return 0.5


def extract_letter_a_to_d(text: str) -> str | None:
    """
    Try to extract a single multiple-choice letter (A–D) from free-form text.
    Priority:
    1) Explicit markers like "Answer: B", "Final answer: C", "Option D", "Choice A"
    2) Line-start patterns like "B) ..." or "C. ..."
    3) Parenthetical markers like "(A)"
    Returns uppercase letter or None if not found.
    """
    import re
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    # 1) Explicit markers
    m = re.search(r"(?i)\b(?:final\s*answer|answer|choice|option)\s*[:\-]?\s*([ABCD])\b", t)
    if m:
        return m.group(1).upper()
    # 2) Line-start patterns (look at first non-empty line)
    first_line = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"\s*([ABCD])\s*[)\.\-]\s*", first_line)
    if m:
        return m.group(1).upper()
    # 3) Parenthetical markers anywhere
    m = re.search(r"\(([ABCD])\)", t)
    if m:
        return m.group(1).upper()
    # 4) Fallback: standalone letter token near the end
    tokens = re.findall(r"\b([ABCD])\b", t)
    if tokens:
        return tokens[-1].upper()
    return None


def extract_one_sentence(text: str) -> str:
    """
    Extract a concise single sentence from free-form text.
    Heuristics:
    - Remove code fences and excessive whitespace
    - Use first sentence ending with . ! ? ; fallback to first ~25 words
    """
    import re
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text
    # Remove fenced code blocks
    t = re.sub(r"```[\s\S]*?```", " ", t)
    # Collapse whitespace
    t = " ".join(t.split())
    # Find first sentence terminator
    m = re.search(r"(.+?[\.\!\?])\s", t + " ")
    if m:
        return m.group(1).strip()
    # Fallback: first ~25 words
    words = t.split()
    if len(words) > 25:
        return " ".join(words[:25]).rstrip(".,;:!?") + "."
    return t