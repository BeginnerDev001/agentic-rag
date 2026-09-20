"""
BM25 sparse retriever for SEC 10-K filings.

Builds an in-memory BM25Okapi index over the ChromaDB corpus at startup,
then disk-caches it so subsequent runs skip the rebuild (~1-2 min for 11k chunks).

Why BM25 alongside dense?
  Dense embeddings excel at semantic similarity but miss exact-match keywords
  (e.g. specific dollar figures, ticker symbols, item numbers like "Item 7A").
  BM25 captures exact term frequency and is complementary — their fusion via RRF
  consistently outperforms either alone on financial Q&A.
"""

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index" / "bm25_index.pkl"


def _simple_tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer: lowercase, strip punctuation, split on whitespace.
    Avoids NLTK dependency. Good enough for financial text retrieval.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Retriever:
    """
    Sparse BM25 retriever backed by rank_bm25.BM25Okapi.

    Corpus is loaded from ChromaDB (all stored chunks) and indexed at init time
    unless a pre-built pickle is found on disk, in which case it loads in ~1 sec.

    Usage:
        retriever = BM25Retriever()
        results = retriever.retrieve("Apple net income fiscal year 2024", top_k=5)
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        vector_store=None,
    ):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required. Install via: uv add rank-bm25"
            )

        self._BM25Okapi = BM25Okapi
        self.index_path = Path(index_path or DEFAULT_INDEX_PATH)

        # Corpus parallel arrays: chunk metadata + raw texts
        self._corpus_chunks: List[Dict[str, Any]] = []
        self._bm25 = None

        self._load_or_build(vector_store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks using BM25 scoring.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.
            ticker: Optional post-retrieval filter by company ticker.
            fiscal_year: Optional post-retrieval filter by fiscal year.

        Returns:
            List of result dicts (chunk_id, text, score, metadata),
            same schema as DenseRetriever.retrieve().
        """
        if not query.strip() or not self._corpus_chunks:
            return []

        tokens = _simple_tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair (score, index) and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in scored:
            if len(results) >= top_k * 3:  # over-fetch before filtering
                break
            chunk = self._corpus_chunks[idx]
            meta = chunk.get("metadata", {})

            # Optional metadata post-filter
            if ticker and meta.get("ticker", "").upper() != ticker.upper():
                continue
            if fiscal_year is not None and meta.get("fiscal_year") != int(fiscal_year):
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
                "distance": 0.0,  # BM25 has no distance concept
                "metadata": meta,
                "retrieval_method": "bm25",
            })

            if len(results) >= top_k:
                break

        return results

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_chunks)

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    def _load_or_build(self, vector_store=None) -> None:
        """Load index from disk cache if available, otherwise build from ChromaDB."""
        if self.index_path.exists():
            print(f"[BM25] Loading cached index from {self.index_path}")
            self._load_from_disk()
        else:
            print("[BM25] No cached index found — building from ChromaDB...")
            self._build_from_vector_store(vector_store)

    def _load_from_disk(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._corpus_chunks = data["corpus_chunks"]
        self._bm25 = data["bm25"]
        print(f"[BM25] Loaded index: {len(self._corpus_chunks)} chunks")

    def _build_from_vector_store(self, vector_store=None) -> None:
        """Dump all chunks from ChromaDB and build a BM25 index."""
        from src.retrieval.vector_store import VectorStore

        vs = vector_store or VectorStore()
        total = vs.count()
        if total == 0:
            raise RuntimeError("ChromaDB is empty — run ingestion first.")

        print(f"[BM25] Fetching {total} chunks from ChromaDB...")
        # ChromaDB get() without query — fetch entire corpus
        raw = vs.collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )

        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])

        self._corpus_chunks = []
        tokenized_corpus = []

        for i, (chunk_id, text, meta) in enumerate(zip(ids, docs, metas)):
            self._corpus_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
            })
            tokenized_corpus.append(_simple_tokenize(text))

        print(f"[BM25] Building BM25Okapi index over {len(tokenized_corpus)} documents...")
        self._bm25 = self._BM25Okapi(tokenized_corpus)

        # Persist to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"corpus_chunks": self._corpus_chunks, "bm25": self._bm25}, f)

        print(f"[BM25] Index saved to {self.index_path}")

    def rebuild(self, vector_store=None) -> None:
        """Force rebuild of BM25 index (deletes existing cache)."""
        if self.index_path.exists():
            self.index_path.unlink()
        self._build_from_vector_store(vector_store)
