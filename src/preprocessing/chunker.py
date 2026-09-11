"""
Structure-aware chunker for parsed SEC 10-K filing sections.

Strategy:
- Tables are kept as atomic chunks (never split across boundaries).
- Long narrative sections are split along sentence boundaries with
  configurable target size and overlap.
- Every chunk inherits full metadata from its parent section for
  downstream retrieval filtering and source citation.

Why structure-aware instead of fixed-size?
  Financial 10-Ks are ~60-70% tables. A naive 500-token chunker
  randomly slices through table rows, separating "Net Income" from
  its dollar value. Structure-aware chunking keeps each table intact
  and respects section boundaries so retrieval recall stays high.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------
# Configuration defaults
# --------------------------------------------------

DEFAULT_TARGET_TOKENS = 512      # ideal chunk size in *word-approx* tokens
DEFAULT_MAX_TOKENS = 1024        # hard ceiling before we force-split
DEFAULT_OVERLAP_TOKENS = 64      # overlap at chunk boundaries


# --------------------------------------------------
# Data model
# --------------------------------------------------

@dataclass
class Chunk:
    """A single retrieval-ready text chunk with full provenance."""

    chunk_id: str
    text: str
    token_count: int

    # Provenance from parent section
    company: str
    ticker: str
    cik: str
    filing_type: str
    fiscal_year: int
    filing_date: Optional[str]
    report_date: Optional[str]
    accession_number: Optional[str]

    # Section context
    section: str
    section_id: str
    part: Optional[str]
    content_type: str            # "narrative" or "table"

    # Chunk position
    chunk_index: int             # 0-based index within this section
    total_chunks: int            # how many chunks this section produced

    # Optional source info
    source_url: Optional[str] = None
    raw_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dictionary for JSONL output."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "company": self.company,
            "ticker": self.ticker,
            "cik": self.cik,
            "filing_type": self.filing_type,
            "fiscal_year": self.fiscal_year,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "accession_number": self.accession_number,
            "section": self.section,
            "section_id": self.section_id,
            "part": self.part,
            "content_type": self.content_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_url": self.source_url,
            "raw_path": self.raw_path,
        }


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _approx_token_count(text: str) -> int:
    """Fast word-level token estimate (close enough for chunking)."""
    return len(text.split())


def _make_chunk_id(ticker: str, fiscal_year: int, section_id: str, idx: int, text: str = "") -> str:
    """Deterministic, unique chunk ID including text hash."""
    content_key = f"{ticker}_{fiscal_year}_{section_id}_{idx}_{text[:128]}"
    short_hash = hashlib.md5(content_key.encode("utf-8")).hexdigest()[:8]
    return f"{ticker}_{fiscal_year}_{section_id}_{idx:03d}_{short_hash}"


_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\(])'   # split after sentence-ending punctuation
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex heuristic."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


# --------------------------------------------------
# Core chunker
# --------------------------------------------------

class SECChunker:
    """
    Structure-aware chunker for SEC 10-K filing sections.

    Rules:
    1. Tables -> always kept as ONE atomic chunk (even if large).
    2. Short narratives (<= target_tokens) -> one chunk.
    3. Long narratives -> split on sentence boundaries with overlap.
    """

    def __init__(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ---------- public API ----------

    def chunk_section(self, section: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a single parsed section dict (one line of the JSONL).
        Returns a list of Chunk objects.
        """
        text = (section.get("text") or "").strip()
        if not text:
            return []

        content_type = section.get("content_type", "narrative")

        # --- Tables: atomic, never split ---
        if content_type == "table":
            raw_chunks = [text]
        # --- Narratives: split if too long ---
        elif _approx_token_count(text) <= self.target_tokens:
            raw_chunks = [text]
        else:
            raw_chunks = self._split_narrative(text)

        # Build Chunk objects with full metadata
        total = len(raw_chunks)
        ticker = section.get("ticker", "UNK")
        fiscal_year = section.get("fiscal_year", 0)
        section_id = section.get("section_id", "unknown")

        chunks: List[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=_make_chunk_id(ticker, fiscal_year, section_id, idx, chunk_text),
                text=chunk_text,
                token_count=_approx_token_count(chunk_text),
                company=section.get("company", ""),
                ticker=ticker,
                cik=section.get("cik", ""),
                filing_type=section.get("filing_type", "10-K"),
                fiscal_year=fiscal_year,
                filing_date=section.get("filing_date"),
                report_date=section.get("report_date"),
                accession_number=section.get("accession_number"),
                section=section.get("section", ""),
                section_id=section_id,
                part=section.get("part"),
                content_type=content_type,
                chunk_index=idx,
                total_chunks=total,
                source_url=section.get("source_url"),
                raw_path=section.get("raw_path"),
            )
            chunks.append(chunk)

        return chunks

    def chunk_filing(self, sections: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk all sections of a single filing."""
        all_chunks: List[Chunk] = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section))
        return all_chunks

    # ---------- internal ----------

    def _split_narrative(self, text: str) -> List[str]:
        """
        Split a long narrative into chunks along sentence boundaries.
        Uses a greedy accumulator: keep adding sentences until we
        exceed target_tokens, then start a new chunk with overlap.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _approx_token_count(sentence)

            # If a single sentence is huge, take it as its own chunk
            if sent_tokens > self.max_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                chunks.append(sentence)
                current_sentences = []
                current_tokens = 0
                continue

            # Would adding this sentence exceed target?
            if current_tokens + sent_tokens > self.target_tokens and current_sentences:
                # Flush current chunk
                chunks.append(" ".join(current_sentences))

                # Start new chunk with overlap from tail of previous
                overlap_sents: List[str] = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    s_tokens = _approx_token_count(s)
                    if overlap_count + s_tokens > self.overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_count += s_tokens

                current_sentences = overlap_sents + [sentence]
                current_tokens = sum(_approx_token_count(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks
