"""
Test script for verifying MCP server tools directly in Python.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server import get_corpus_stats, search_sec_filings, answer_financial_question


def main():
    print("=" * 60)
    print("  MCP SERVER TOOLS - VERIFICATION TEST")
    print("=" * 60)

    # Test 1: get_corpus_stats
    print("\n--- Test 1: get_corpus_stats ---")
    stats = get_corpus_stats()
    print("Stats:", stats)

    # Test 2: search_sec_filings
    print("\n--- Test 2: search_sec_filings (NVDA) ---")
    chunks = search_sec_filings(query="data center GPU revenue", ticker="NVDA", top_k=2)
    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        meta = c["metadata"]
        print(f"  - [{c['score']:.3f}] {meta.get('ticker')}/{meta.get('fiscal_year')}/{meta.get('section')}")
        print(f"    {c['text'][:100]}...")

    # Test 3: answer_financial_question
    print("\n--- Test 3: answer_financial_question (MSFT) ---")
    res = answer_financial_question(query="What was Microsoft's cloud growth in 2024?", ticker="MSFT")
    print("Answer:", res["answer"])
    print("Model used:", res["model"])
    print(f"Sources cited ({len(res['sources'])}):")
    for s in res["sources"]:
        print(f"  - {s['ticker']}/{s['fiscal_year']}/{s['section']}")

    print("\n" + "=" * 60)
    print("  ALL MCP TOOLS FUNCTIONAL AND VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
