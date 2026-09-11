"""
Single-process batch ingestion script for ChromaDB vector store.
Loads sentence-transformers model once and ingests all chunks sequentially.
"""

import os

# Prevent OpenMP / MKL multi-threading crashes on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
torch.set_num_threads(1)

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


CHUNKS_DIR = Path("data/processed/chunks")


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into ChromaDB vector store.")
    parser.add_argument("--clear", action="store_true", help="Clear vector collection before ingesting")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    args = parser.parse_args()

    print("=" * 60)
    print("CHROMADB VECTOR STORE INGESTION (BATCH ENGINE)")
    print("=" * 60)
    sys.stdout.flush()

    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore()

    if args.clear:
        print("Clearing vector collection...")
        vector_store.clear()
        sys.stdout.flush()

    chunk_files = sorted(CHUNKS_DIR.glob("*/*_chunks.jsonl"))
    print(f"Found {len(chunk_files)} chunk files")
    print(f"Initial DB count: {vector_store.count()} vectors")
    print()
    sys.stdout.flush()

    start_time = time.time()
    total_added = 0

    for idx, cf in enumerate(chunk_files, 1):
        ticker = cf.parent.name
        chunks = []
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line.strip()))

        if not chunks:
            continue

        # Check how many are already ingested by loading first chunk_id
        # Batch ingest
        for b in range(0, len(chunks), args.batch_size):
            batch = chunks[b : b + args.batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embedder.embed_texts(texts, batch_size=len(texts))
            added = vector_store.add_chunks(batch, embeddings)
            total_added += added

        print(f"[{idx:02d}/{len(chunk_files)}] {ticker}/{cf.name} | Ingested: {len(chunks)} | DB Count: {vector_store.count()}")
        sys.stdout.flush()

    elapsed = round(time.time() - start_time, 2)
    print()
    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Files processed:       {len(chunk_files)}")
    print(f"Total chunks ingested: {total_added}")
    print(f"Final DB vector count: {vector_store.count()}")
    print(f"Execution time:        {elapsed}s")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
