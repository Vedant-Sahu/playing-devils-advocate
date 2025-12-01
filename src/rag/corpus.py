"""Corpus loading and filtering for physics RAG.

Loads and filters physics content from:
1. LibreTexts - open textbook content
2. arXiv - physics paper abstracts
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Dict, Any, Optional
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Physics-related arXiv category prefixes
ARXIV_PHYSICS_CATEGORIES = {
    "physics",      # General physics
    "quant-ph",     # Quantum physics
    "hep-th",       # High energy physics - theory
    "hep-ph",       # High energy physics - phenomenology
    "hep-ex",       # High energy physics - experiment
    "hep-lat",      # High energy physics - lattice
    "cond-mat",     # Condensed matter
    "astro-ph",     # Astrophysics
    "gr-qc",        # General relativity and quantum cosmology
    "nucl-th",      # Nuclear theory
    "nucl-ex",      # Nuclear experiment
    "math-ph",      # Mathematical physics
    "nlin",         # Nonlinear sciences
}

# LibreTexts physics-related source patterns
LIBRETEXTS_PHYSICS_PATTERNS = [
    "phys.libretexts",
    "physics",
    "quantum",
    "mechanics",
    "electr",  # electricity, electromagnetism, etc.
    "thermo",
    "optics",
    "relativity",
]


@dataclass
class Document:
    """A document chunk for RAG."""
    content: str
    metadata: Dict[str, Any]
    source: str  # "libretexts" or "arxiv"


def load_libretexts_physics(max_docs: Optional[int] = None) -> List[Document]:
    """Load physics-related content from LibreTexts dataset.
    
    Args:
        max_docs: Maximum number of documents to load (None for all)
        
    Returns:
        List of Document objects
    """
    logger.info("Loading LibreTexts dataset...")
    
    try:
        ds = load_dataset(
            "common-pile/libretexts", 
            split="train",
            trust_remote_code=True,  # Some datasets require this
        )
    except Exception as e:
        logger.error(f"Failed to load LibreTexts dataset: {e}")
        logger.exception(e)
        return []
    
    documents = []
    count = 0
    
    for example in ds:
        # Get the URL from metadata
        metadata = example.get("metadata", {}) or {}
        url = (metadata.get("url") or "").lower()
        text = example.get("text") or ""
        title = (metadata.get("title") or "").lower()
        
        # Check if physics-related based on URL or title
        is_physics = (
            "phys.libretexts" in url or
            any(pattern in url for pattern in LIBRETEXTS_PHYSICS_PATTERNS) or
            any(pattern in title for pattern in LIBRETEXTS_PHYSICS_PATTERNS)
        )
        
        if is_physics and text.strip():
            doc = Document(
                content=text.strip(),
                metadata={
                    "source_url": url or example.get("source") or "",
                    "title": metadata.get("title") or "",
                    "license": metadata.get("license") or "unknown",
                },
                source="libretexts"
            )
            documents.append(doc)
            count += 1
            
            if max_docs and count >= max_docs:
                break
    
    logger.info(f"Loaded {len(documents)} physics documents from LibreTexts")
    return documents


def load_arxiv_physics(max_docs: Optional[int] = None) -> List[Document]:
    """Load physics paper abstracts from arXiv dataset.
    
    Args:
        max_docs: Maximum number of documents to load (None for all)
        
    Returns:
        List of Document objects
    """
    logger.info("Loading arXiv dataset (streaming)...")
    
    try:
        # Use streaming to avoid downloading entire dataset
        ds = load_dataset(
            "arxiv-community/arxiv_dataset", 
            split="train", 
            streaming=True,
            trust_remote_code=True,  # Required for this dataset
        )
    except Exception as e:
        logger.error(f"Failed to load arXiv dataset: {e}")
        return []
    
    documents = []
    count = 0
    processed = 0
    
    for example in ds:
        processed += 1
        if processed % 100000 == 0:
            logger.info(f"Processed {processed} arXiv papers, found {count} physics papers...")
        
        # Check if physics-related based on categories
        categories = example.get("categories", "").split()
        is_physics = any(
            cat.split(".")[0] in ARXIV_PHYSICS_CATEGORIES 
            for cat in categories
        )
        
        if is_physics:
            title = example.get("title", "").strip()
            abstract = example.get("abstract", "").strip()
            
            if abstract:
                # Combine title and abstract for richer context
                content = f"Title: {title}\n\nAbstract: {abstract}"
                
                doc = Document(
                    content=content,
                    metadata={
                        "arxiv_id": example.get("id", ""),
                        "title": title,
                        "authors": example.get("authors", ""),
                        "categories": example.get("categories", ""),
                        "update_date": example.get("update_date", ""),
                    },
                    source="arxiv"
                )
                documents.append(doc)
                count += 1
                
                if max_docs and count >= max_docs:
                    break
    
    logger.info(f"Loaded {len(documents)} physics abstracts from arXiv")
    return documents


def load_physics_corpus(
    include_libretexts: bool = True,
    include_arxiv: bool = True,
    max_libretexts: Optional[int] = None,
    max_arxiv: Optional[int] = 50000,  # Default limit for arXiv (there are many)
) -> List[Document]:
    """Load combined physics corpus from all sources.
    
    Args:
        include_libretexts: Whether to include LibreTexts content
        include_arxiv: Whether to include arXiv abstracts
        max_libretexts: Max documents from LibreTexts
        max_arxiv: Max documents from arXiv
        
    Returns:
        Combined list of Document objects
    """
    documents = []
    
    if include_libretexts:
        documents.extend(load_libretexts_physics(max_docs=max_libretexts))
    
    if include_arxiv:
        documents.extend(load_arxiv_physics(max_docs=max_arxiv))
    
    logger.info(f"Total corpus size: {len(documents)} documents")
    return documents
