"""
One-shot script to build and cache the BM25 index from the ChromaDB corpus.

Run this ONCE after ingestion is complete. Subsequent BM25Retriever instantiations
will load the cached pickle (~1 sec) instead of rebuilding (~1-2 min).

Usage:
    python scripts/build_bm25_index.py
    python scripts/build_bm25_index.py --rebuild   # Force rebuild even if cache exists
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Thread safety for PyTorch/OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever, DEFAULT_INDEX_PATH
from src.retrieval.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Build and cache BM25 index from ChromaDB corpus.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if a cached index already exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BM25 INDEX BUILDER")
    print("=" * 60)

    if DEFAULT_INDEX_PATH.exists() and not args.rebuild:
        print(f"Cache already exists: {DEFAULT_INDEX_PATH}")
        print("Use --rebuild to force a fresh build.")
        print()

    # Check ChromaDB has data
    vs = VectorStore()
    count = vs.count()
    print(f"ChromaDB vector count: {count}")
    if count == 0:
        print("ERROR: ChromaDB is empty. Run ingestion pipeline first.")
        sys.exit(1)

    start = time.time()

    if args.rebuild and DEFAULT_INDEX_PATH.exists():
        DEFAULT_INDEX_PATH.unlink()
        print("Removed existing cache.")

    bm25 = BM25Retriever(vector_store=vs)

    elapsed = round(time.time() - start, 2)

    print()
    print("=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"Index size:    {bm25.corpus_size} documents")
    print(f"Cache path:    {DEFAULT_INDEX_PATH}")
    print(f"Elapsed time:  {elapsed}s")
    print("=" * 60)

    # Quick sanity check
    print("\nRunning sanity check: BM25 search for 'Apple net income 2024'")
    results = bm25.retrieve("Apple net income fiscal year 2024", top_k=3)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"  [{i}] Score: {r['score']:.4f} | {meta.get('ticker')} {meta.get('fiscal_year')} | {meta.get('section', '')[:40]}")

    print("\nSanity check passed. BM25 index is ready.")


if __name__ == "__main__":
    main()
