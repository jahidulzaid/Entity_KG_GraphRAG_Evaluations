"""Document loading and processing utilities for KG-RAG approaches."""

import json
import os
import pickle
from typing import Any

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter


class DocumentLoader:
    """Loads and processes documents for knowledge graph construction."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 24,
        file_extension: str = ".pdf",
        file_extensions: list[str] | None = None,
    ):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            file_extension: File extension to filter for
            file_extensions: Optional list of file extensions to allow
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if file_extensions is not None:
            extensions = file_extensions
        else:
            extensions = [file_extension]
        self.file_extensions = {ext.lower() for ext in extensions}
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _flatten_json(self, data: Any, prefix: str = "") -> list[str]:
        lines: list[str] = []
        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten_json(value, next_prefix))
            return lines
        if isinstance(data, list):
            for index, value in enumerate(data):
                next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                lines.extend(self._flatten_json(value, next_prefix))
            return lines
        if data is None:
            return lines
        value_text = str(data).strip()
        if not value_text:
            return lines
        if prefix:
            lines.append(f"{prefix}: {value_text}")
        else:
            lines.append(value_text)
        return lines

    def _load_text_content(self, file_path: str) -> list[Document]:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

        raw_documents = [Document(page_content=text, metadata={"source": file_path})]
        split_documents = self.text_splitter.split_documents(raw_documents)
        return filter_complex_metadata(split_documents)

    def _load_json_content(self, file_path: str) -> list[Document]:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON {file_path}: {str(e)}")
            return []

        text = "\n".join(self._flatten_json(data))
        raw_documents = [Document(page_content=text, metadata={"source": file_path})]
        split_documents = self.text_splitter.split_documents(raw_documents)
        return filter_complex_metadata(split_documents)

    def load_document(self, file_path: str) -> list[Document]:
        """
        Load and process a single document.

        Args:
            file_path: Path to the document

        Returns
        -------
            List of processed document chunks
        """
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".pdf":
                raw_documents = PyPDFLoader(file_path=file_path).load()
                split_documents = self.text_splitter.split_documents(raw_documents)
                return filter_complex_metadata(split_documents)
            if extension == ".txt":
                return self._load_text_content(file_path)
            if extension == ".json":
                return self._load_json_content(file_path)
            print(f"Skipping unsupported file type {extension}: {file_path}")
            return []
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def load_directory(
        self, directory_path: str, file_filter: str | None = None
    ) -> list[Document]:
        """
        Load and process all documents in a directory.

        Args:
            directory_path: Path to the directory
            file_filter: Optional string to filter filenames (e.g., "AAPL")

        Returns
        -------
            List of processed document chunks from all files
        """
        documents = []

        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            # Apply filters
            extension = os.path.splitext(filename)[1].lower()
            if extension not in self.file_extensions:
                continue

            if file_filter and file_filter not in filename:
                continue

            # Construct full file path
            file_path = os.path.join(directory_path, filename)

            # Load and process the file
            processed_docs = self.load_document(file_path)
            documents.extend(processed_docs)

            print(f"Processed: {filename} - {len(processed_docs)} chunks")

        print(f"Total documents processed: {len(documents)}")
        return documents


def load_documents(
    directory_path: str,
    file_filter: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 24,
    pickle_path: str | None = None,
    file_extensions: list[str] | None = None,
) -> list[Document]:
    """
    Load documents from a directory or a pickle file.

    Args:
        directory_path: Path to the directory containing documents
        file_filter: Optional string to filter filenames
        chunk_size: Size of chunks for text splitting
        chunk_overlap: Overlap between chunks
        pickle_path: Optional path to a pickle file containing pre-chunked documents
        file_extensions: Optional list of file extensions to load

    Returns
    -------
        List of processed document chunks
    """
    # If pickle path is provided and exists, load from pickle
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            documents = pickle.load(f)
            if isinstance(documents, list):
                return documents
            raise ValueError("Invalid pickle file format")
    else:
        # Otherwise load from directory
        loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_extensions=file_extensions,
        )
        return loader.load_directory(directory_path, file_filter)


def load_graph_documents(file_path: str) -> Any:
    """
    Load graph documents from a pickle file.

    Args:
        file_path: Path to the graph documents pickle file

    Returns
    -------
        Loaded graph documents
    """
    with open(file_path, "rb") as f:
        graph_documents = pickle.load(f)
    print(f"Graph documents loaded from {file_path}")
    return graph_documents
