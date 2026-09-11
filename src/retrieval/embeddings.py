import os
from typing import List, Optional, Union
import numpy as np


DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating normalized vector embeddings.

    Supports:
    - Lazy loading of model weights
    - Batch encoding for large chunk sets
    - Single query encoding for retrieval
    - Embedding normalization for exact cosine similarity dot products
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is required. "
                    "Install it via `uv add sentence-transformers`."
                )

            # Set PyTorch CPU thread parallelism to 1 to avoid Windows C++ DLL conflict
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            # Determine vector dimension
            dummy_vec = self._model.encode("test", convert_to_numpy=True)
            self._dimension = int(dummy_vec.shape[0])
        return self._model

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension  # type: ignore

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """
        Embed a list of text strings into a list of float vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model encoding.
            show_progress_bar: Whether to display tqdm progress.

        Returns:
            List of float lists representing normalized vector embeddings.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query string.

        Args:
            query: The user search query.

        Returns:
            Float list representing the query vector.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding.tolist()
