# app/embedder.py

"""
Embedding module: loads the embedding model and provides text embedding functionality.
"""

import torch
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.
        """
        self.model = SentenceTransformer(model_name, device="mps" if torch.cuda.is_available() else "cpu")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vector representations.

        Args:
            texts (List[str]): List of input strings.

        Returns:
            np.ndarray: 2D array of embedding vectors (num_texts, dim).
        """
        return self.model.encode(texts, convert_to_numpy=True)
