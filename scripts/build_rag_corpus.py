#!/usr/bin/env python3
"""Build the physics RAG corpus from LibreTexts and arXiv.

This script downloads physics content from:
1. LibreTexts (open textbooks)
2. arXiv (physics paper abstracts)

And builds a FAISS vector store for retrieval.

Usage:
    python scripts/build_rag_corpus.py [OPTIONS]

Options:
    --max-libretexts INT    Max documents from LibreTexts (default: all)
    --max-arxiv INT         Max documents from arXiv (default: 50000)
    --chunk-size INT        Text chunk size (default: 500)
    --force                 Force rebuild even if exists
    --libretexts-only       Only include LibreTexts (faster for testing)
    --arxiv-only            Only include arXiv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.rag.corpus import load_physics_corpus
from src.rag.vectorstore import build_vectorstore, DEFAULT_VECTORSTORE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build physics RAG corpus from LibreTexts and arXiv"
    )
    parser.add_argument(
        "--max-libretexts",
        type=int,
        default=None,
        help="Maximum documents from LibreTexts (default: all)"
    )
    parser.add_argument(
        "--max-arxiv",
        type=int,
        default=50000,
        help="Maximum documents from arXiv (default: 50000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size for splitting (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap (default: 50)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if vector store exists"
    )
    parser.add_argument(
        "--libretexts-only",
        action="store_true",
        help="Only include LibreTexts (faster for testing)"
    )
    parser.add_argument(
        "--arxiv-only",
        action="store_true",
        help="Only include arXiv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: {DEFAULT_VECTORSTORE_PATH})"
    )
    
    args = parser.parse_args()
    
    # Check if vector store already exists
    output_path = Path(args.output) if args.output else DEFAULT_VECTORSTORE_PATH
    if output_path.exists() and not args.force:
        logger.info(f"Vector store already exists at {output_path}")
        logger.info("Use --force to rebuild")
        return
    
    # Determine which sources to include
    include_libretexts = not args.arxiv_only
    include_arxiv = not args.libretexts_only
    
    if not include_libretexts and not include_arxiv:
        logger.error("Cannot specify both --libretexts-only and --arxiv-only")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Building Physics RAG Corpus")
    logger.info("=" * 60)
    logger.info(f"Include LibreTexts: {include_libretexts}")
    logger.info(f"Include arXiv: {include_arxiv}")
    logger.info(f"Max LibreTexts docs: {args.max_libretexts or 'all'}")
    logger.info(f"Max arXiv docs: {args.max_arxiv}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 60)
    
    # Load corpus
    logger.info("\n[1/2] Loading physics corpus...")
    documents = load_physics_corpus(
        include_libretexts=include_libretexts,
        include_arxiv=include_arxiv,
        max_libretexts=args.max_libretexts,
        max_arxiv=args.max_arxiv,
    )
    
    if not documents:
        logger.error("No documents loaded! Check your internet connection.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(documents)} documents total")
    
    # Count by source
    libretexts_count = sum(1 for d in documents if d.source == "libretexts")
    arxiv_count = sum(1 for d in documents if d.source == "arxiv")
    logger.info(f"  - LibreTexts: {libretexts_count}")
    logger.info(f"  - arXiv: {arxiv_count}")
    
    # Build vector store
    logger.info("\n[2/2] Building vector store...")
    logger.info("This will embed all documents using OpenAI API (may take a while)...")
    
    try:
        vectorstore = build_vectorstore(
            documents=documents,
            save_path=output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! Vector store built and saved.")
        logger.info(f"Location: {output_path}")
        logger.info("")
        logger.info("To enable RAG in the pipeline, set:")
        logger.info("  export USE_RAG=1")
        logger.info("or add USE_RAG=1 to your .env file")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
