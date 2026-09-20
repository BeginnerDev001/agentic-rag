"""
Smoke test for all Phase 3 retrieval components:
  BM25Retriever, HybridRetriever, CrossEncoderReranker

Verifies that each module can be imported, initialized, and queried successfully.

Usage:
    python scripts/test_hybrid_retrieval.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_QUERY = "What was Apple's total net sales in fiscal year 2024?"
TOP_K = 5


def test_dense():
    print("\n" + "=" * 50)
    print("TEST 1: DenseRetriever (baseline)")
    print("=" * 50)
    from src.retrieval.retriever import DenseRetriever
    r = DenseRetriever()
    print(f"  Corpus size: {r.corpus_size}")
    t0 = time.time()
    results = r.retrieve(TEST_QUERY, top_k=TOP_K, ticker="AAPL")
    elapsed = time.time() - t0
    print(f"  Retrieved {len(results)} chunks in {elapsed:.2f}s")
    for i, res in enumerate(results, 1):
        m = res["metadata"]
        print(f"  [{i}] score={res['score']:.4f} | {m.get('ticker')} {m.get('fiscal_year')} | {m.get('section','')[:40]}")
    print("  PASS ✓")
    return results


def test_bm25():
    print("\n" + "=" * 50)
    print("TEST 2: BM25Retriever")
    print("=" * 50)
    from src.retrieval.bm25_retriever import BM25Retriever
    t0 = time.time()
    r = BM25Retriever()
    load_elapsed = time.time() - t0
    print(f"  Index loaded in {load_elapsed:.2f}s | Corpus: {r.corpus_size} docs")
    t0 = time.time()
    results = r.retrieve(TEST_QUERY, top_k=TOP_K, ticker="AAPL")
    elapsed = time.time() - t0
    print(f"  Retrieved {len(results)} chunks in {elapsed:.3f}s")
    for i, res in enumerate(results, 1):
        m = res["metadata"]
        print(f"  [{i}] score={res['score']:.4f} | {m.get('ticker')} {m.get('fiscal_year')} | {m.get('section','')[:40]}")
    print("  PASS ✓")
    return results


def test_hybrid():
    print("\n" + "=" * 50)
    print("TEST 3: HybridRetriever (RRF)")
    print("=" * 50)
    from src.retrieval.hybrid_retriever import HybridRetriever
    t0 = time.time()
    r = HybridRetriever()
    load_elapsed = time.time() - t0
    print(f"  Initialized in {load_elapsed:.2f}s")
    t0 = time.time()
    results = r.retrieve(TEST_QUERY, top_k=TOP_K, ticker="AAPL")
    elapsed = time.time() - t0
    print(f"  Retrieved {len(results)} chunks in {elapsed:.3f}s")
    for i, res in enumerate(results, 1):
        m = res["metadata"]
        print(f"  [{i}] rrf_score={res['score']:.5f} | {m.get('ticker')} {m.get('fiscal_year')} | {m.get('section','')[:40]}")
    print("  PASS ✓")
    return results


def test_hybrid_reranked(hybrid_results=None):
    print("\n" + "=" * 50)
    print("TEST 4: CrossEncoderReranker on Hybrid candidates")
    print("=" * 50)
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    if hybrid_results is None:
        hybrid_results = HybridRetriever().retrieve(TEST_QUERY, top_k=20, ticker="AAPL")

    t0 = time.time()
    reranked = reranker.rerank(TEST_QUERY, hybrid_results, top_k=TOP_K)
    elapsed = time.time() - t0
    print(f"  Reranked {len(hybrid_results)} → top-{TOP_K} in {elapsed:.2f}s")
    for i, res in enumerate(reranked, 1):
        m = res["metadata"]
        print(f"  [{i}] reranker_score={res['reranker_score']:.4f} | {m.get('ticker')} {m.get('fiscal_year')} | {m.get('section','')[:40]}")
    print("  PASS ✓")


def main():
    print("=" * 50)
    print("  PHASE 3 — HYBRID RETRIEVAL SMOKE TESTS")
    print("=" * 50)
    print(f"  Query: '{TEST_QUERY}'")

    test_dense()
    test_bm25()
    hybrid_results = test_hybrid()

    # Fetch more for reranker
    from src.retrieval.hybrid_retriever import HybridRetriever
    candidates = HybridRetriever().retrieve(TEST_QUERY, top_k=20, ticker="AAPL")
    test_hybrid_reranked(candidates)

    print("\n" + "=" * 50)
    print("  ALL PHASE 3 TESTS PASSED ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
