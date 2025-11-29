"""
Web Search utility for retrieving factual information.

Uses Tavily API for search. Set TAVILY_API_KEY in .env.
Free tier: 1000 searches/month.
"""

import os
from typing import Optional

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for information relevant to the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with search results, or empty string if unavailable
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    
    if not api_key:
        return ""
    
    if not TAVILY_AVAILABLE:
        print("    [WebSearch] tavily package not installed. Run: pip install tavily-python")
        return ""
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
        
        # Format results
        parts = []
        
        # Include the AI-generated answer if available
        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}")
        
        # Include top results
        for i, result in enumerate(response.get("results", [])[:max_results], 1):
            title = result.get("title", "")
            content = result.get("content", "")[:500]  # Limit content length
            parts.append(f"[{i}] {title}: {content}")
        
        return "\n\n".join(parts) if parts else ""
        
    except Exception as e:
        print(f"    [WebSearch] Error: {e}")
        return ""


def extract_search_queries(question: str, options: list[str]) -> list[str]:
    """
    Extract potential search queries from a question and its options.
    
    Looks for:
    - Proper nouns (capitalized terms)
    - Technical terms
    - Specific values or constants mentioned
    
    Args:
        question: The question text
        options: List of answer options
        
    Returns:
        List of search queries to run
    """
    import re
    
    queries = []
    
    # Combine question and options for analysis
    full_text = question + " " + " ".join(options)
    
    # Find capitalized phrases (likely proper nouns/technical terms)
    # Match 1-3 consecutive capitalized words
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', full_text)
    
    # Filter out common words and sentence starters
    common_starts = {"The", "A", "An", "This", "That", "Which", "What", "How", "If", "When", "Where"}
    proper_nouns = [p for p in proper_nouns if p.split()[0] not in common_starts]
    
    # Add unique proper nouns as queries
    seen = set()
    for noun in proper_nouns:
        if noun.lower() not in seen and len(noun) > 3:
            queries.append(f"{noun} physics")
            seen.add(noun.lower())
    
    # Limit to top 3 queries
    return queries[:3]
