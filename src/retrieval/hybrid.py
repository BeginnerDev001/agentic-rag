"""
Hybrid Retriever combining Dense Vector Search and BM25 Lexical Search
using Reciprocal Rank Fusion (RRF).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid Retriever combining ChromaDB Dense Vector Search and BM25 Lexical Search.
    Uses Reciprocal Rank Fusion (RRF) to merge search rankings.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using Reciprocal Rank Fusion (RRF) over Dense + BM25 results.

        Args:
            query: User query string.
            top_k: Final number of hybrid chunks to return.
            ticker: Optional ticker filter.
            fiscal_year: Optional fiscal year filter.
            section: Optional section filter.
            candidate_pool_size: Number of top results to retrieve from each retriever before merging.
            rrf_k: Smoothing constant for RRF formula (default 60).
            dense_weight: Relative weight for dense vector rank.
            bm25_weight: Relative weight for BM25 lexical rank.

        Returns:
            List of combined chunk dicts with rrf_score, dense_rank, bm25_rank, text, and metadata.
        """
        # 1. Retrieve candidate lists from both retrievers
        dense_results = self.dense.retrieve(
            query=query,
            top_k=candidate_pool_size,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=candidate_pool_size,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        # 2. Map chunks by unique chunk_id / text hash
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        # Process dense vector results
        for rank, res in enumerate(dense_results, 1):
            cid = res.get("chunk_id") or res.get("id") or str(hash(res.get("text", "")))
            chunk_map[cid] = res
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight * (1.0 / (rrf_k + rank))

        # Process BM25 lexical results
        for rank, res in enumerate(bm25_results, 1):
            cid = res.get("id") or res.get("chunk_id") or str(hash(res.get("text", "")))
            if cid not in chunk_map:
                chunk_map[cid] = res
            bm25_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight * (1.0 / (rrf_k + rank))

        # 3. Sort all candidate chunks by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        merged_results = []
        for cid in sorted_cids[:top_k]:
            orig = chunk_map[cid]
            merged_results.append({
                "chunk_id": cid,
                "text": orig.get("text", ""),
                "rrf_score": round(rrf_scores[cid], 6),
                "dense_rank": dense_ranks.get(cid),
                "bm25_rank": bm25_ranks.get(cid),
                "metadata": orig.get("metadata", {}),
            })

        return merged_results
