"""
Verification test script for Reranker (Cross-Encoder reranking over Hybrid search results).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


def main():
    print("=" * 65)
    print("  CROSS-ENCODER RERANKER - VERIFICATION TEST")
    print("=" * 65)

    print("Initializing HybridRetriever and Cross-Encoder Reranker...")
    hybrid = HybridRetriever()
    reranker = Reranker()

    query = "What were Apple's R&D expenses and iPhone revenue in 2024?"
    print(f"\nQuery: {query}")
    print("\n--- Step 1: Retrieving Top 10 Hybrid Candidates ---")
    candidates = hybrid.retrieve(query, top_k=10, ticker="AAPL", fiscal_year=2024)

    for i, c in enumerate(candidates, 1):
        m = c["metadata"]
        print(f"  Candidate {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] RRF: {c['rrf_score']}")

    print("\n--- Step 2: Reranking candidates down to Top 3 using Cross-Encoder ---")
    reranked = reranker.rerank(query, candidates, top_k=3)

    for i, r in enumerate(reranked, 1):
        m = r["metadata"]
        score_str = f"{r.get('rerank_score'):.4f}" if r.get('rerank_score') is not None else "N/A"
        print(f"  Reranked {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] Score: {score_str}")
        print(f"     Preview: {r['text'][:120]}...")

    print("\n" + "=" * 65)
    print("  RERANKER TEST COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
