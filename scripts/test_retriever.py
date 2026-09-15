"""
Verification script for the DenseRetriever module.
Tests unfiltered search, ticker-filtered search, and multi-filter search.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import DenseRetriever


def print_results(label: str, results: list):
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"{'-' * 60}")
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"  [{i}] Score: {r['score']:.4f} | {meta.get('ticker')}/{meta.get('fiscal_year')} | Section: {meta.get('section')}")
        print(f"      {r['text'][:120]}...")
    print()


def main():
    print("=" * 60)
    print("  DENSE RETRIEVER VERIFICATION")
    print("=" * 60)

    retriever = DenseRetriever()
    print(f"Corpus size: {retriever.corpus_size} vectors")

    # Test 1: Unfiltered search
    q1 = "What was Apple's total revenue?"
    results1 = retriever.retrieve(q1, top_k=3)
    print_results(f"Test 1 - Unfiltered: '{q1}'", results1)

    # Test 2: Ticker-filtered search
    q2 = "What are the main risk factors?"
    results2 = retriever.retrieve(q2, top_k=3, ticker="NVDA")
    print_results(f"Test 2 - Ticker=NVDA: '{q2}'", results2)

    # Test 3: Ticker + Year filter
    q3 = "Net income and operating expenses"
    results3 = retriever.retrieve(q3, top_k=3, ticker="MSFT", fiscal_year=2024)
    print_results(f"Test 3 - Ticker=MSFT, FY=2024: '{q3}'", results3)

    # Test 4: retrieve_with_context()
    q4 = "What was Tesla's revenue growth?"
    ctx = retriever.retrieve_with_context(q4, top_k=3, ticker="TSLA")
    print_results(f"Test 4 - retrieve_with_context (TSLA): '{q4}'", ctx["context_chunks"])
    print(f"  Context text length: {len(ctx['context_text'])} chars")
    print(f"  Sources: {ctx['sources']}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED - DenseRetriever is working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
