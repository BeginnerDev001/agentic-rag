"""
Test script for verifying BM25Retriever keyword search and metadata filtering.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever


def main():
    print("=" * 60)
    print("  BM25 LEXICAL RETRIEVER - VERIFICATION TEST")
    print("=" * 60)

    print("Indexing corpus chunks...")
    retriever = BM25Retriever()
    print(f"Indexed {retriever.num_docs} chunks across SEC filings.")

    # Test 1: Exact metric keyword query
    q1 = "research and development expense 2024"
    print(f"\n--- Test 1: Unfiltered Keyword Query ---")
    print(f"Query: {q1}")
    res1 = retriever.retrieve(q1, top_k=3)
    for i, r in enumerate(res1, 1):
        m = r["metadata"]
        print(f"  {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] Score: {r['score']}")
        print(f"     Preview: {r['text'][:120]}...")

    # Test 2: Filtered query for NVDA
    q2 = "datacenter revenue blackwell"
    print(f"\n--- Test 2: Filtered Query (NVDA 2024) ---")
    print(f"Query: {q2}")
    res2 = retriever.retrieve(q2, top_k=3, ticker="NVDA", fiscal_year=2024)
    for i, r in enumerate(res2, 1):
        m = r["metadata"]
        print(f"  {i}. [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] Score: {r['score']}")
        print(f"     Preview: {r['text'][:120]}...")

    print("\n" + "=" * 60)
    print("  BM25 RETRIEVER TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
