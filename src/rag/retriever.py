"""Physics knowledge retriever for RAG-augmented teacher.

This module provides a simple interface for retrieving relevant physics
content to augment the teacher's explanations.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

# Lazy import to avoid dependency chain issues
# from .vectorstore import load_vectorstore, DEFAULT_VECTORSTORE_PATH

logger = logging.getLogger(__name__)

# Environment variable to enable/disable RAG
RAG_ENABLED_ENV = "USE_RAG"


@dataclass
class RetrievedContext:
    """Retrieved context for teacher augmentation."""
    chunks: List[str]
    sources: List[str]
    
    def format_for_prompt(self, max_chunks: int = 5) -> str:
        """Format retrieved context for inclusion in teacher prompt.
        
        Args:
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted string for prompt injection
        """
        if not self.chunks:
            return ""
        
        chunks_to_use = self.chunks[:max_chunks]
        
        formatted_parts = []
        for i, chunk in enumerate(chunks_to_use, 1):
            # Truncate very long chunks
            if len(chunk) > 800:
                chunk = chunk[:800] + "..."
            formatted_parts.append(f"[{i}] {chunk}")
        
        return "\n\n".join(formatted_parts)
    
    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return len(self.chunks) > 0


class PhysicsRetriever:
    """Retriever for physics knowledge from the vector store."""
    
    def __init__(self, vectorstore=None, k: int = 5):
        """Initialize the retriever.
        
        Args:
            vectorstore: FAISS vector store (loads from disk if None)
            k: Number of documents to retrieve per query
        """
        self.k = k
        self._vectorstore = vectorstore
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazily initialize the vector store."""
        if self._initialized:
            return self._vectorstore is not None
        
        self._initialized = True
        
        if self._vectorstore is None:
            from .vectorstore import load_vectorstore
            self._vectorstore = load_vectorstore()
        
        if self._vectorstore is None:
            logger.warning(
                "RAG vector store not found. Run 'python scripts/build_rag_corpus.py' "
                "to build the physics corpus."
            )
            return False
        
        return True
    
    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievedContext:
        """Retrieve relevant physics content for a query.
        
        Args:
            query: The question or topic to retrieve context for
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            RetrievedContext with chunks and source information
        """
        if not self._ensure_initialized():
            return RetrievedContext(chunks=[], sources=[])
        
        k = k or self.k
        
        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            
            chunks = [doc.page_content for doc in docs]
            sources = [
                doc.metadata.get("source_url") or doc.metadata.get("arxiv_id") or "unknown"
                for doc in docs
            ]
            
            logger.debug(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
            return RetrievedContext(chunks=chunks, sources=sources)
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RetrievedContext(chunks=[], sources=[])
    
    def retrieve_for_question(self, question: str, k: Optional[int] = None) -> str:
        """Retrieve and format context for a GPQA question.
        
        Convenience method that returns formatted text ready for prompt injection.
        
        Args:
            question: The GPQA question text
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string (empty string if no context found)
        """
        context = self.retrieve(question, k=k)
        return context.format_for_prompt()


# Global retriever instance (lazy initialization)
_retriever: Optional[PhysicsRetriever] = None


def get_retriever() -> Optional[PhysicsRetriever]:
    """Get the global physics retriever instance.
    
    Returns None if RAG is disabled via environment variable.
    
    Returns:
        PhysicsRetriever instance or None if RAG disabled
    """
    global _retriever
    
    # Check if RAG is enabled
    rag_enabled = os.getenv(RAG_ENABLED_ENV, "").lower() in ("1", "true", "yes")
    
    if not rag_enabled:
        logger.debug("RAG is disabled (set USE_RAG=1 to enable)")
        return None
    
    if _retriever is None:
        _retriever = PhysicsRetriever()
    
    return _retriever


def retrieve_physics_context(question: str, k: int = 5) -> str:
    """Convenience function to retrieve physics context for a question.
    
    This is the main entry point for the teacher agent.
    Returns empty string if RAG is disabled or unavailable.
    
    Args:
        question: The question to retrieve context for
        k: Number of chunks to retrieve
        
    Returns:
        Formatted context string for prompt injection
    """
    retriever = get_retriever()
    
    if retriever is None:
        return ""
    
    return retriever.retrieve_for_question(question, k=k)
