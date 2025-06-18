# app/document_loader.py

from pathlib import Path
from typing import List

def load_documents_txt(folder_path: str) -> List[str]:
    """
    Load plain text documents from a folder.

    Args:
        folder_path (str): Path to folder containing .txt documents.

    Returns:
        List[str]: A list of raw text contents.
    """
    # Compute absolute path relative to this file's location
    base_dir = Path(__file__).resolve().parent.parent  # Go to project root
    docs_path = base_dir / folder_path

    documents = []
    for file_path in docs_path.glob("*.txt"):
        with file_path.open("r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents


def split_documents(documents: List[str], chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split each document into smaller chunks with optional overlap.

    Args:
        documents (List[str]): List of full documents as strings.
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap
    return chunks
