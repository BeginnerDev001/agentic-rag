"""
Dense retriever module for SEC 10-K RAG pipeline.

Wraps EmbeddingModel + VectorStore into a single retrieval interface
with optional metadata filtering (ticker, fiscal_year, section).
"""

from typing import Any, Dict, List, Optional

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


class DenseRetriever:
    """
    High-level retrieval interface for SEC 10-K filings.

    Embeds a natural-language query and searches the ChromaDB vector store
    for the most semantically similar chunks. Supports metadata filtering
    to narrow results by company, fiscal year, or filing section.

    Usage:
        retriever = DenseRetriever()
        results = retriever.retrieve("What was Apple's net income in 2024?", top_k=5, ticker="AAPL")
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()

    @property
    def corpus_size(self) -> int:
        """Return total number of chunks in the vector store."""
        return self.vector_store.count()

    def _build_where_filter(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB metadata filter dict from optional constraints.

        ChromaDB uses:
          - Single condition:  {"ticker": "AAPL"}
          - Multiple conditions: {"$and": [{"ticker": "AAPL"}, {"fiscal_year": 2024}]}
        """
        conditions: List[Dict[str, Any]] = []

        if ticker:
            conditions.append({"ticker": ticker.upper()})
        if fiscal_year is not None:
            conditions.append({"fiscal_year": int(fiscal_year)})
        if section:
            conditions.append({"section": section})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a natural-language query.

        Args:
            query: Natural-language question or search phrase.
            top_k: Number of top results to return (default 5).
            ticker: Filter by company ticker symbol (e.g. "AAPL").
            fiscal_year: Filter by fiscal year (e.g. 2024).
            section: Filter by 10-K section ID (e.g. "item_7").

        Returns:
            List of result dicts, each containing:
              - chunk_id: Unique identifier for the chunk
              - text: Full chunk text content
              - score: Cosine similarity score (0.0 - 1.0)
              - distance: Raw cosine distance
              - metadata: Dict with ticker, fiscal_year, section, content_type, etc.
        """
        if not query or not query.strip():
            return []

        # 1. Embed the query into a dense vector
        query_vector = self.embedder.embed_query(query.strip())

        # 2. Build optional metadata filter
        where_filter = self._build_where_filter(
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        # 3. Search ChromaDB vector store
        results = self.vector_store.query(
            query_embedding=query_vector,
            n_results=top_k,
            where_filter=where_filter,
        )

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and package them into a structured context dict
        ready for the answer generator.

        Returns:
            Dict containing:
              - query: Original query string
              - num_results: Number of chunks retrieved
              - context_chunks: List of result dicts (same as retrieve())
              - context_text: Concatenated text from all retrieved chunks
              - sources: List of citation source dicts (ticker, year, section, chunk_id)
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        # Build concatenated context string for the LLM
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
            })

        return {
            "query": query,
            "num_results": len(results),
            "context_chunks": results,
            "context_text": "\n\n".join(context_parts),
            "sources": sources,
        }
