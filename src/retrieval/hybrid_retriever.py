"""
Hybrid Retriever using Reciprocal Rank Fusion (RRF) of Dense + BM25 results.

RRF formula (Cormack et al. 2009):
    score(doc) = Σ  1 / (k + rank_i(doc))
    where k=60 is an empirically strong default.

Why RRF over score normalization?
  Dense scores (cosine 0-1) and BM25 scores (unbounded, corpus-dependent) live in
  different spaces — direct sum/average is misleading. RRF depends only on rank
  position, making it robust to scale differences and requiring zero tuning.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search and BM25 sparse search
    via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("Apple net income 2024", top_k=5, ticker="AAPL")
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
        dense_top_k_multiplier: int = 3,
    ):
        """
        Args:
            dense_retriever: Pre-instantiated DenseRetriever (or creates one).
            bm25_retriever: Pre-instantiated BM25Retriever (or creates one).
            rrf_k: RRF smoothing constant (default 60 is near-universally optimal).
            dense_top_k_multiplier: Over-fetch factor for each sub-retriever before fusion.
                                    E.g. if top_k=5 → each retriever fetches 15 candidates.
        """
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        self.dense_top_k_multiplier = dense_top_k_multiplier

    @property
    def corpus_size(self) -> int:
        return self.dense.corpus_size

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k chunks by fusing dense and BM25 rankings via RRF.

        Args:
            query: Natural-language search query.
            top_k: Final number of results to return.
            ticker: Optional metadata filter by ticker.
            fiscal_year: Optional metadata filter by fiscal year.
            section: Optional metadata filter by section (dense only).

        Returns:
            List of result dicts with RRF fusion score, same schema as DenseRetriever.
            Each result includes a 'retrieval_method' key set to 'hybrid'.
        """
        if not query.strip():
            return []

        fetch_k = top_k * self.dense_top_k_multiplier

        # -- Fetch from both retrievers --
        dense_results = self.dense.retrieve(
            query=query,
            top_k=fetch_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=fetch_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
        )

        # -- RRF Fusion --
        fused = self._rrf_fuse(dense_results, bm25_results)

        return fused[:top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve and package results into context dict for the answer generator.
        Drop-in replacement for DenseRetriever.retrieve_with_context().
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        context_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            source_label = f"[{meta.get('ticker', '?')}/{meta.get('fiscal_year', '?')}/{meta.get('section', '?')}]"
            context_parts.append(f"--- Source {i} {source_label} ---\n{r['text']}")
            sources.append({
                "chunk_id": r["chunk_id"],
                "ticker": meta.get("ticker"),
                "fiscal_year": meta.get("fiscal_year"),
                "section": meta.get("section"),
                "score": r["score"],
                "retrieval_method": r.get("retrieval_method", "hybrid"),
            })

        return {
            "query": query,
            "num_results": len(results),
            "context_chunks": results,
            "context_text": "\n\n".join(context_parts),
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # RRF internals
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        For each unique chunk_id, accumulate:
            rrf_score += 1 / (k + rank)
        across both lists. Then sort descending by rrf_score.
        """
        k = self.rrf_k
        rrf_scores: Dict[str, float] = {}
        chunk_registry: Dict[str, Dict[str, Any]] = {}  # chunk_id → chunk dict

        for rank, result in enumerate(dense_results, start=1):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_registry:
                chunk_registry[cid] = result

        for rank, result in enumerate(bm25_results, start=1):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_registry:
                chunk_registry[cid] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused: List[Dict[str, Any]] = []
        for cid in sorted_ids:
            chunk = dict(chunk_registry[cid])  # copy
            chunk["score"] = rrf_scores[cid]
            chunk["retrieval_method"] = "hybrid"
            fused.append(chunk)

        return fused
