"""
Cross-Encoder Reranker for SEC 10-K RAG pipeline.

Uses a cross-encoder model to score each (query, chunk_text) pair jointly —
unlike bi-encoder retrieval which scores them independently. This gives much
higher precision at the cost of latency (no ANN index — O(n) inference).

Typical usage in pipeline:
    1. HybridRetriever fetches top-20 candidates (fast, coarse)
    2. CrossEncoderReranker scores all 20 pairs and returns top-5 (slow, precise)

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~22M params, CPU-friendly (~0.5s for 20 pairs on a laptop CPU).
  Trained on MS-MARCO passage ranking — strong zero-shot generalization.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHUNK_CHARS = 512  # truncate chunk text fed to cross-encoder to control latency


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.

    Lazy-loads the model on first call. Can be used standalone or composed
    after any retriever (DenseRetriever, BM25Retriever, HybridRetriever).

    Usage:
        retriever = HybridRetriever()
        reranker = CrossEncoderReranker()

        candidates = retriever.retrieve(query, top_k=20)
        reranked   = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = None  # lazy-load

    @property
    def model(self):
        """Lazy-load the CrossEncoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers>=2.0 is required for CrossEncoder. "
                    "Install via: uv add sentence-transformers"
                )
            print(f"[Reranker] Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            print("[Reranker] Model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each (query, chunk_text) pair with the cross-encoder and re-sort.

        Args:
            query: The user's natural-language question.
            chunks: List of chunk dicts (from any retriever — must have 'text' key).
            top_k: How many top results to return. None = return all, re-sorted.

        Returns:
            Re-sorted list of chunk dicts with an added 'reranker_score' key.
            The 'score' field is overwritten with the cross-encoder logit.
        """
        if not chunks or not query.strip():
            return chunks

        # Build pairs list for batch scoring
        pairs = [
            (query, chunk["text"][:MAX_CHUNK_CHARS])
            for chunk in chunks
        ]

        # Batch predict — returns array of logits (higher = more relevant)
        scores = self.model.predict(pairs)

        # Attach reranker score and sort
        reranked = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["reranker_score"] = float(score)
            chunk_copy["score"] = float(score)  # overwrite retrieval score
            chunk_copy["retrieval_method"] = chunk.get("retrieval_method", "unknown") + "+reranked"
            reranked.append(chunk_copy)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def retrieve_and_rerank(
        self,
        query: str,
        retriever,
        top_k: int = 5,
        fetch_k: int = 20,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: retrieve fetch_k candidates then rerank to top_k.

        Args:
            query: User query.
            retriever: Any retriever with a .retrieve() method.
            top_k: Final number of results to return.
            fetch_k: Number of candidates to fetch before reranking.
            ticker, fiscal_year, section: Metadata filters passed to retriever.

        Returns:
            Top-k re-ranked chunk dicts.
        """
        candidates = retriever.retrieve(
            query=query,
            top_k=fetch_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )
        return self.rerank(query, candidates, top_k=top_k)
