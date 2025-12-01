"""Vector store management for physics RAG.

Uses FAISS for local vector storage (no external server needed).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional

from .corpus import Document, load_physics_corpus

# Lazy imports to avoid dependency issues
# These are imported inside functions that need them

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "data" / "physics_vectorstore"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _convert_to_langchain_docs(documents: List[Document]):
    """Convert our Document objects to LangChain Document objects."""
    from langchain_core.documents import Document as LangchainDocument
    return [
        LangchainDocument(
            page_content=doc.content,
            metadata={**doc.metadata, "source_type": doc.source}
        )
        for doc in documents
    ]


def _simple_text_splitter(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Simple text splitter that doesn't require heavy dependencies.
    
    Splits on paragraph boundaries, then sentences, then words.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Try splitting on paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself is too long, split by sentences
            if len(para) > chunk_size:
                sentences = para.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|')
                current_chunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = current_chunk + " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If sentence is still too long, just truncate
                        current_chunk = sent[:chunk_size] if len(sent) > chunk_size else sent
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add overlap by including end of previous chunk at start of next
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][-chunk_overlap:] if len(chunks[i-1]) > chunk_overlap else chunks[i-1]
            overlapped_chunks.append(prev_end + " " + chunks[i])
        chunks = overlapped_chunks
    
    return chunks


def build_vectorstore(
    documents: Optional[List[Document]] = None,
    save_path: Optional[Path] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    """Build a FAISS vector store from physics documents.
    
    Args:
        documents: List of Document objects (loads corpus if None)
        save_path: Where to save the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: OpenAI embedding model to use
        
    Returns:
        FAISS vector store
    """
    save_path = save_path or DEFAULT_VECTORSTORE_PATH
    
    # Load corpus if not provided
    # Lazy imports
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    if documents is None:
        logger.info("Loading physics corpus...")
        documents = load_physics_corpus()
    
    if not documents:
        raise ValueError("No documents to build vector store from")
    
    # Split documents into chunks using simple splitter
    logger.info(f"Splitting {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    from langchain_core.documents import Document as LangchainDocument
    
    chunks = []
    for doc in documents:
        text_chunks = _simple_text_splitter(doc.content, chunk_size, chunk_overlap)
        for chunk_text in text_chunks:
            chunks.append(LangchainDocument(
                page_content=chunk_text,
                metadata={**doc.metadata, "source_type": doc.source}
            ))
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Create embeddings with rate limiting
    logger.info(f"Creating embeddings with {embedding_model}...")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        chunk_size=100,  # Smaller batches to avoid rate limits
    )
    
    # Build FAISS index in batches to avoid rate limits
    logger.info("Building FAISS index (this may take a while)...")
    import time
    
    batch_size = 500  # Process 500 chunks at a time
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
        
        # Rate limit: wait between batches
        if i + batch_size < len(chunks):
            time.sleep(2)  # 2 second pause between batches
    
    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving vector store to {save_path}...")
    vectorstore.save_local(str(save_path))
    
    logger.info("Vector store built successfully!")
    return vectorstore


def load_vectorstore(
    path: Optional[Path] = None,
    embedding_model: str = "text-embedding-3-small",
) -> Optional[FAISS]:
    """Load a saved FAISS vector store.
    
    Args:
        path: Path to the saved vector store
        embedding_model: OpenAI embedding model (must match what was used to build)
        
    Returns:
        FAISS vector store or None if not found
    """
    path = path or DEFAULT_VECTORSTORE_PATH
    
    # Lazy imports
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    if not path.exists():
        logger.warning(f"Vector store not found at {path}")
        return None
    
    logger.info(f"Loading vector store from {path}...")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    # allow_dangerous_deserialization is needed for FAISS
    vectorstore = FAISS.load_local(  # noqa: F821
        str(path), 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    logger.info("Vector store loaded successfully!")
    return vectorstore


def get_or_build_vectorstore(
    path: Optional[Path] = None,
    force_rebuild: bool = False,
    **build_kwargs
) -> FAISS:
    """Get existing vector store or build a new one.
    
    Args:
        path: Path to the vector store
        force_rebuild: If True, rebuild even if exists
        **build_kwargs: Arguments passed to build_vectorstore
        
    Returns:
        FAISS vector store
    """
    path = path or DEFAULT_VECTORSTORE_PATH
    
    if not force_rebuild and path.exists():
        vectorstore = load_vectorstore(path)
        if vectorstore is not None:
            return vectorstore
    
    return build_vectorstore(save_path=path, **build_kwargs)
