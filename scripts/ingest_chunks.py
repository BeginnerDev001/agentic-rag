"""
Ingest all pre-chunked SEC 10-K filings into ChromaDB vector store.
"""

import os
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
    parser.add_argument("--ticker", type=str, help="Filter ingestion by ticker symbol")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding & DB insert")
    parser.add_argument("--clear", action="store_true", help="Clear vector collection before ingesting")
    args = parser.parse_args()

    print("=" * 60)
    print("CHROMADB VECTOR STORE INGESTION")
    print("=" * 60)
    sys.stdout.flush()

    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore()

    if args.clear:
        print("Clearing vector store collection...")
        vector_store.clear()
        print(f"Collection reset. Current count: {vector_store.count()}")
        sys.stdout.flush()

    chunk_files = sorted(CHUNKS_DIR.glob("*/*_chunks.jsonl"))
    if args.ticker:
        t = args.ticker.upper()
        chunk_files = [f for f in chunk_files if f.parent.name == t]

    print(f"Found {len(chunk_files)} chunk files to ingest")
    print(f"Initial DB count: {vector_store.count()} vectors")
    print("-" * 60)
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

        file_added = 0
        for b in range(0, len(chunks), args.batch_size):
            batch = chunks[b : b + args.batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embedder.embed_texts(texts, batch_size=len(texts))
            added = vector_store.add_chunks(batch, embeddings)
            file_added += added

        total_added += file_added
        print(f"[{idx:02d}/{len(chunk_files)}] {ticker}/{cf.name} | Chunks: {len(chunks)} | DB Total: {vector_store.count()}")
        sys.stdout.flush()

    elapsed = round(time.time() - start_time, 2)
    print("=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)
    print(f"Files processed:       {len(chunk_files)}")
    print(f"Total chunks ingested: {total_added}")
    print(f"Final DB vector count: {vector_store.count()}")
    print(f"Total execution time:  {elapsed}s")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
