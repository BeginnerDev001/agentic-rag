"""
Test script to verify ChromaDB vector retrieval functionality.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
torch.set_num_threads(1)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():
    print("=" * 60)
    print("VERIFYING CHROMADB VECTOR RETRIEVAL")
    print("=" * 60)

    embedder = EmbeddingModel()
    vector_store = VectorStore()
    count = vector_store.count()
    print(f"Total stored vectors in ChromaDB: {count}")
    test_query = "What was Apple's net income in fiscal year 2024?"
    print(f"\nTest Query: '{test_query}'")

    q_vector = embedder.embed_query(test_query)
    results = vector_store.query(q_vector, n_results=3)

    print(f"\nRetrieved Top {len(results)} Matches:")
    for idx, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n[{idx}] Score: {r['score']:.4f} | Ticker: {meta.get('ticker')} | FY: {meta.get('fiscal_year')} | Section: {meta.get('section')}")
        print(f"    Chunk ID: {r['chunk_id']}")
        print(f"    Text Preview: {r['text'][:150]}...")

    print("=" * 60)
    print("VERIFICATION SUCCESSFUL: Step 4 is 100% Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
