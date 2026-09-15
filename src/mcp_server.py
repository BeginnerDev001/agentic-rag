"""
MCP Server for SEC 10-K Financial RAG System.

Exposes vector search and answer generation as Model Context Protocol (MCP) tools
for external agents, Claude Desktop, and IDE integrations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU single-thread for PyTorch/OpenMP safety on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from mcp.server.fastmcp import FastMCP

from src.generation.answer import NaiveRAGGenerator
from src.retrieval.retriever import DenseRetriever

# Initialize FastMCP server
mcp = FastMCP("SEC-10K-Financial-RAG")

# Lazy-loaded singletons for efficiency
_retriever: Optional[DenseRetriever] = None
_generator: Optional[NaiveRAGGenerator] = None


def get_retriever() -> DenseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever()
    return _retriever


def get_generator() -> NaiveRAGGenerator:
    global _generator
    if _generator is None:
        _generator = NaiveRAGGenerator(retriever=get_retriever())
    return _generator


@mcp.tool()
def search_sec_filings(
    query: str,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search 10,525 pre-processed SEC 10-K filing chunks using vector similarity.

    Args:
        query: Natural language search query or financial topic (e.g. "AI risks and opportunities").
        ticker: Optional stock ticker filter (e.g. "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA").
        fiscal_year: Optional fiscal year filter (e.g. 2024, 2023, 2022).
        section: Optional 10-K section filter (e.g. "Item 1A. Risk Factors", "Item 7", "Item 8").
        top_k: Number of relevant chunks to retrieve (default 5).

    Returns:
        List of matching chunk dicts with similarity scores, text, and source metadata.
    """
    retriever = get_retriever()
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
    )
    return results


@mcp.tool()
def answer_financial_question(
    query: str,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a grounded, cited answer to a financial question using SEC 10-K filings.

    Args:
        query: Financial question (e.g. "What were Microsoft's major revenue drivers in 2024?").
        ticker: Optional stock ticker filter (e.g. "MSFT").
        fiscal_year: Optional fiscal year filter (e.g. 2024).
        section: Optional 10-K section filter.

    Returns:
        Dict containing:
          - answer: Grounded text answer with [TICKER/YEAR/SECTION] citations.
          - sources: List of cited filing chunk metadata.
          - model: LLM model used for synthesis.
    """
    generator = get_generator()
    result = generator.answer(
        query=query,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
    )
    return result


@mcp.tool()
def get_corpus_stats() -> Dict[str, Any]:
    """
    Get statistics about the underlying SEC 10-K vector database.

    Returns:
        Dict with total chunk count and status.
    """
    retriever = get_retriever()
    return {
        "total_vectors": retriever.corpus_size,
        "status": "active",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "distance_metric": "cosine",
    }


def main():
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
