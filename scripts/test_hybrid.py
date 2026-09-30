"""
Verification test script for HybridRetriever (Reciprocal Rank Fusion over Dense + BM25).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever


def main():
    print("=" * 65)
    print("  HYBRID RETRIEVER (RRF) - VERIFICATION TEST")
    print("=" * 65)

    print("Initializing HybridRetriever (Dense + BM25)...")
    hybrid = HybridRetriever()

    # Query 1: Unfiltered hybrid query
    q1 = "What was Microsoft's Azure cloud revenue growth in 2024?"
    print(f"\n--- Test 1: Hybrid Search (MSFT 2024) ---")
    print(f"Query: {q1}")
    res1 = hybrid.retrieve(q1, top_k=3, ticker="MSFT", fiscal_year=2024)

    for i, r in enumerate(res1, 1):
        m = r["metadata"]
        d_rank = r["dense_rank"] if r["dense_rank"] is not None else "-"
        b_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
        print(f"  {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] RRF Score: {r['rrf_score']} (Dense Rank: {d_rank}, BM25 Rank: {b_rank})")
        print(f"     Preview: {r['text'][:120]}...")

    # Query 2: Exact numeric / technical keyword query
    q2 = "NVIDIA Blackwell GPU architecture datacenter revenue"
    print(f"\n--- Test 2: Keyword-Heavy Search (NVDA 2024) ---")
    print(f"Query: {q2}")
    res2 = hybrid.retrieve(q2, top_k=3, ticker="NVDA", fiscal_year=2024)

    for i, r in enumerate(res2, 1):
        m = r["metadata"]
        d_rank = r["dense_rank"] if r["dense_rank"] is not None else "-"
        b_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
        print(f"  {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] RRF Score: {r['rrf_score']} (Dense Rank: {d_rank}, BM25 Rank: {b_rank})")
        print(f"     Preview: {r['text'][:120]}...")

    print("\n" + "=" * 65)
    print("  HYBRID RETRIEVER TEST COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
