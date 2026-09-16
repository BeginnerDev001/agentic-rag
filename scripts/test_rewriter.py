"""
Test script for verifying QueryRewriter (rewrite and decompose methods).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.rewriter import QueryRewriter


def main():
    print("=" * 60)
    print("  QUERY REWRITER & DECOMPOSER - VERIFICATION TEST")
    print("=" * 60)

    rewriter = QueryRewriter()

    # Test 1: Simple query rewrite
    q1 = "What money did Apple make from selling iPhones in 2023?"
    r1 = rewriter.rewrite(q1)
    print("\n--- Test 1: Rephrase to SEC Terminology ---")
    print(f"Original:  {q1}")
    print(f"Rewritten: {r1}")

    # Test 2: Rewrite with feedback
    q2 = "What are Microsoft's main risk factors?"
    fb2 = "Previous search returned generic disclaimers. Need specific legal and AI risk factors."
    r2 = rewriter.rewrite(q2, feedback=fb2)
    print("\n--- Test 2: Rewrite with Feedback ---")
    print(f"Original:  {q2}")
    print(f"Feedback:  {fb2}")
    print(f"Rewritten: {r2}")

    # Test 3: Multi-company query decomposition
    q3 = "Compare the 2024 total net revenue of Microsoft, Apple, and NVIDIA."
    d3 = rewriter.decompose(q3)
    print("\n--- Test 3: Query Decomposition ---")
    print(f"Original Multi-Part Query: {q3}")
    print(f"Decomposed Sub-queries ({len(d3)}):")
    for idx, sub_q in enumerate(d3, 1):
        print(f"  {idx}. {sub_q}")

    print("\n" + "=" * 60)
    print("  QUERY REWRITER TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
