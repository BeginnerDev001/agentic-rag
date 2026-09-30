"""
Cross-Encoder Reranker for deep query-chunk relevance scoring.

Uses SentenceTransformers CrossEncoder to re-score and re-rank candidate chunks
retrieved via Hybrid Search.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prevent OpenMP / PyTorch Windows process crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """
    Cross-Encoder Reranker for high-precision passage scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name, device=device)
            except Exception as e:
                print(f"[Reranker] Warning: Could not initialize CrossEncoder ({e}). Fallback to rank scoring.")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a candidate list of chunks using the CrossEncoder model.

        Args:
            query: The input user query.
            chunks: Candidate chunk dictionaries (from Dense, BM25, or Hybrid retrieval).
            top_k: Number of reranked chunks to return.

        Returns:
            List of top-k chunks with added 'rerank_score'.
        """
        if not chunks:
            return []

        if not query.strip():
            return chunks[:top_k]

        # If CrossEncoder model is available, compute pair scores
        if self.model is not None:
            pairs = [[query, chunk.get("text", "")[:512]] for chunk in chunks]
            try:
                scores = self.model.predict(pairs)
                reranked_chunks = []
                for idx, chunk in enumerate(chunks):
                    c_copy = dict(chunk)
                    score_val = float(scores[idx])
                    c_copy["rerank_score"] = score_val
                    c_copy["reranker_score"] = score_val
                    c_copy["score"] = score_val
                    reranked_chunks.append(c_copy)

                reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                return reranked_chunks[:top_k]
            except Exception as e:
                print(f"[Reranker] Prediction failed: {e}. Returning candidate chunks.")

        # Fallback if CrossEncoder is not available
        for idx, chunk in enumerate(chunks):
            score_val = float(chunk.get("rrf_score", chunk.get("score", 1.0 / (idx + 1))))
            chunk["rerank_score"] = score_val
            chunk["reranker_score"] = score_val

        return chunks[:top_k]


# Alias for backward compatibility
CrossEncoderReranker = Reranker
