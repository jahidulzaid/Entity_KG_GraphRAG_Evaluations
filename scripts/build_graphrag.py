#!/usr/bin/env python
"""Script to build GraphRAG artifacts from documents."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.graphrag_based.indexer import GraphRAGIndexer
from kg_rag.utils.document_loader import load_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build GraphRAG artifacts from documents"
    )
    parser.add_argument(
        "--docs-dir", type=str, required=True, help="Directory containing the documents"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Directory to save artifacts"
    )
    parser.add_argument(
        "--artifacts-name",
        type=str,
        default="graphrag_artifacts",
        help="Name of the artifacts file",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache", help="Directory for caching"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="vector_stores",
        help="Directory for vector stores",
    )
    parser.add_argument(
        "--file-filter",
        type=str,
        default=None,
        help="Optional string to filter filenames",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        default=".pdf",
        help="Comma-separated list of file extensions to load (default: .pdf)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=24, help="Overlap between document chunks"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o", help="LLM model to use"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def main():
    """Build GraphRAG artifacts from documents."""
    # Load environment variables
    load_dotenv()

    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_path = output_dir / f"{args.artifacts_name}.pkl"
    cache_dir = resolve_output_path(output_dir, args.cache_dir)
    vector_store_dir = resolve_output_path(output_dir, args.vector_store_dir)

    print(f"Loading documents from {args.docs_dir}...")
    documents = load_documents(
        directory_path=args.docs_dir,
        file_filter=args.file_filter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_extensions=parse_file_extensions(args.file_extensions),
    )
    print(f"Loaded {len(documents)} document chunks")

    indexer = GraphRAGIndexer(
        cache_dir=str(cache_dir),
        vector_store_dir=str(vector_store_dir),
        artifacts_dir=str(output_dir),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose,
    )

    artifacts = indexer.index_documents(documents)
    indexer.save_artifacts(artifacts, str(artifacts_path))

    print("\nGraphRAG artifacts built successfully")
    print(f"Artifacts saved to {artifacts_path}")
    print(f"Vector store directory: {vector_store_dir}")
    print(f"Cache directory: {cache_dir}")


def resolve_output_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def parse_file_extensions(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    return [item if item.startswith(".") else f".{item}" for item in items]


if __name__ == "__main__":
    main()
