"""
Test script for verifying AgenticRAG self-correcting orchestrator.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG


def main():
    print("=" * 60)
    print("  AGENTIC RAG ORCHESTRATOR - VERIFICATION TEST")
    print("=" * 60)

    agent = AgenticRAG(max_retries=2)

    # Test 1: Single question with self-correction
    q1 = "What were Meta's primary AI investment risk factors in 2024?"
    print(f"\n--- Test 1: Single Query ---")
    print(f"Question: {q1}")
    res1 = agent.ask(q1, ticker="META", fiscal_year=2024)

    print(f"\nStatus:  {res1['status']}")
    print(f"Retries: {res1['retries']}")
    print(f"Answer:\n{res1['answer']}")
    print(f"Sources ({len(res1['sources'])}):")
    for s in res1["sources"]:
        print(f"  - {s['ticker']}/{s['fiscal_year']}/{s['section']}")

    print("\nExecution Trace:")
    for t in res1["execution_trace"]:
        print(f"  Attempt {t['attempt']}: Rewritten -> '{t['rewritten_query']}' | Outcome -> {t.get('outcome')}")

    # Test 2: Multi-part decomposed query
    q2 = "Compare 2024 AI risk factors for Microsoft and Meta."
    print(f"\n--- Test 2: Decomposed Multi-Company Query ---")
    print(f"Question: {q2}")
    res2 = agent.ask(q2)

    print(f"\nStatus: {res2['status']}")
    if "sub_queries" in res2:
        print(f"Sub-queries ({len(res2['sub_queries'])}):")
        for sq in res2["sub_queries"]:
            print(f"  - {sq}")

    print(f"Combined Answer:\n{res2['answer']}")

    print("\n" + "=" * 60)
    print("  AGENTIC RAG TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
