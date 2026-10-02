"""
BM25 Lexical Retriever for SEC 10-K text chunks.

Provides pure-Python, zero-dependency BM25Okapi keyword search over
processed filing chunks with metadata filtering.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    BM25Okapi Lexical Search Engine over SEC 10-K text chunks.
    """

    def __init__(
        self,
        chunks_dir: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks_dir = chunks_dir or (PROJECT_ROOT / "data" / "processed" / "chunks")
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0

        self._load_and_index()

    def _load_and_index(self):
        """Load chunks recursively from disk and compute BM25 corpus statistics."""
        if not self.chunks_dir.exists():
            return

        chunks_list = []
        # Support both json and jsonl files in ticker subdirectories
        for file_path in self.chunks_dir.rglob("*"):
            if file_path.suffix in (".json", ".jsonl") and file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix == ".jsonl":
                            for line in f:
                                if line.strip():
                                    chunks_list.append(json.loads(line.strip()))
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                chunks_list.extend(data)
                            elif isinstance(data, dict):
                                chunks_list.append(data)
                except Exception:
                    continue

        self.chunks = chunks_list
        self.num_docs = len(self.chunks)
        if self.num_docs == 0:
            return

        total_len = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.get("text", ""))
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avgdl = total_len / self.num_docs if self.num_docs > 0 else 1.0

        # Precompute IDFs using standard Okapi BM25 formula
        for token, freq in self.doc_freqs.items():
            idf_val = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[token] = max(0.0, idf_val)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k matching chunks using BM25Okapi scoring.

        Args:
            query: Keyword or phrase query string.
            top_k: Number of results to return.
            ticker: Optional ticker filter (e.g. "AAPL").
            fiscal_year: Optional fiscal year filter (e.g. 2024).
            section: Optional section filter (e.g. "item_7").

        Returns:
            List of result dicts containing text, bm25_score, chunk_id, and metadata.
        """
        if not query.strip() or self.num_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored_results = []

        for idx, chunk in enumerate(self.chunks):
            # Extract metadata from root keys or nested 'metadata' dict
            meta = chunk.get("metadata", {})
            chunk_ticker = chunk.get("ticker") or meta.get("ticker", "")
            chunk_year = chunk.get("fiscal_year") or meta.get("fiscal_year")
            chunk_section = chunk.get("section") or meta.get("section", "")
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "")

            # Apply metadata filters
            if ticker and str(chunk_ticker).upper() != str(ticker).upper():
                continue
            if fiscal_year is not None and str(chunk_year) != str(fiscal_year):
                continue
            if section and str(chunk_section).lower() != str(section).lower():
                continue

            doc_tokens = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]
            if doc_len == 0:
                continue

            tf_counter = Counter(doc_tokens)
            score = 0.0

            for q_tok in query_tokens:
                if q_tok not in tf_counter:
                    continue
                tf = tf_counter[q_tok]
                idf_val = self.idf.get(q_tok, 0.0)

                # BM25 term score calculation
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += idf_val * (numerator / denominator)

            if score > 0.0:
                scored_results.append({
                    "id": chunk_id,
                    "text": chunk.get("text", ""),
                    "score": round(score, 4),
                    "metadata": {
                        "ticker": chunk_ticker,
                        "fiscal_year": chunk_year,
                        "section": chunk_section,
                        "company": chunk.get("company", meta.get("company", "")),
                    },
                })

        # Sort descending by BM25 score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
