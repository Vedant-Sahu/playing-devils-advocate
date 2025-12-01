"""RAG (Retrieval-Augmented Generation) module for physics knowledge retrieval.

This module is separate from the main pipeline and can be enabled/disabled
via the USE_RAG environment variable.

Usage:
    # Enable RAG by setting environment variable
    export USE_RAG=1
    
    # Or in Python
    from src.rag.retriever import retrieve_physics_context
    context = retrieve_physics_context("What is quantum entanglement?")
"""

# Lazy imports to avoid dependency issues
# Import these directly when needed:
#   from src.rag.retriever import PhysicsRetriever, get_retriever, retrieve_physics_context
#   from src.rag.corpus import load_physics_corpus
#   from src.rag.vectorstore import build_vectorstore

__all__ = ["PhysicsRetriever", "get_retriever", "retrieve_physics_context"]


def __getattr__(name):
    """Lazy import to avoid dependency chain issues."""
    if name in ("PhysicsRetriever", "get_retriever", "retrieve_physics_context"):
        from .retriever import PhysicsRetriever, get_retriever, retrieve_physics_context
        return {"PhysicsRetriever": PhysicsRetriever, 
                "get_retriever": get_retriever,
                "retrieve_physics_context": retrieve_physics_context}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
