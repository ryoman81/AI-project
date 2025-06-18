# FAISS vector database construction and querying
# app/vector_store.py

from typing import List, Tuple
import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self, dim: int):
        """
        Initializes a FAISS index for dense vector search.

        Args:
            dim (int): Dimension of the embeddings.
        """
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []  # raw texts for retrieval
        self.embeddings = []  # store for persistence

    def add_documents(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Add embedded vectors and corresponding texts to the store.

        Args:
            embeddings (np.ndarray): 2D array of embedding vectors (num_docs, dim).
            texts (List[str]): Corresponding list of text segments.
        """
        vectors = embeddings.astype('float32')
        self.index.add(vectors)
        self.documents.extend(texts)
        self.embeddings.extend([v for v in vectors])

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[dict]:
        """
        Perform similarity search with query vector.

        Args:
            query_embedding (np.ndarray): Query vector (1D or 2D).
            k (int): Number of top results to return.

        Returns:
            List[dict]: List of {"text": ..., "score": ...} dicts.
        """
        if query_embedding.ndim == 1:
            query_vector = query_embedding.astype('float32').reshape(1, -1)
        else:
            query_vector = query_embedding.astype('float32')
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, score in zip(indices[0], distances[0]):
            if i < len(self.documents):
                results.append({
                    "text": self.documents[i],
                    "score": float(score)
                })

        return results

    def save(self, path: str) -> None:
        """
        Save FAISS index to disk.

        Args:
            path (str): File path to save index.
        """
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

    def load(self, path: str) -> None:
        """
        Load FAISS index from disk.

        Args:
            path (str): File path to load index from.
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
