"""
End-to-end test for the Naive RAG Generator.
Tests the full pipeline: Question -> Retriever -> LLM -> Cited Answer.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer import NaiveRAGGenerator


def main():
    print("=" * 60)
    print("  NAIVE RAG GENERATOR - END-TO-END TEST")
    print("=" * 60)

    generator = NaiveRAGGenerator(top_k=3, max_tokens=256)
    print(f"Model: {generator.model}")
    print(f"Corpus: {generator.retriever.corpus_size} vectors")

    # Test 1: Factual question with ticker filter
    print("\n" + "-" * 60)
    print("  Test 1: Factual question (AAPL)")
    print("-" * 60)
    q1 = "What was Apple's total net sales in fiscal year 2024?"
    result1 = generator.answer(q1, ticker="AAPL")
    print(f"Q: {result1['query']}")
    print(f"A: {result1['answer']}")
    print(f"Sources: {len(result1['sources'])} chunks used")

    time.sleep(3)

    # Test 2: Multi-company question (unfiltered)
    print("\n" + "-" * 60)
    print("  Test 2: Unfiltered cross-company question")
    print("-" * 60)
    q2 = "What are the main risk factors related to AI mentioned in recent SEC filings?"
    result2 = generator.answer(q2)
    print(f"Q: {result2['query']}")
    print(f"A: {result2['answer']}")
    print(f"Sources: {len(result2['sources'])} chunks used")

    time.sleep(3)

    # Test 3: Question likely not in corpus
    print("\n" + "-" * 60)
    print("  Test 3: Out-of-corpus question")
    print("-" * 60)
    q3 = "What will be Tesla's stock price in 2030?"
    result3 = generator.answer(q3, ticker="TSLA")
    print(f"Q: {result3['query']}")
    print(f"A: {result3['answer']}")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
