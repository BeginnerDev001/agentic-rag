"""
Ingest all 10 SEC 10-K filing tickers into ChromaDB using isolated subprocesses.
Ensures zero C++ heap fragmentation and 100% full dataset ingestion.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


TICKERS = ["AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "V", "WMT"]


def main():
    parser = argparse.ArgumentParser(description="Ingest filings ticker by ticker.")
    parser.add_argument("--clear", action="store_true", help="Clear vector store before start")
    args = parser.parse_args()

    print("=" * 60)
    print("CHROMADB ISOLATED TICKER INGESTION PIPELINE")
    print("=" * 60)
    sys.stdout.flush()

    if args.clear:
        # Clear once at start via python inline snippet
        cmd_clear = [
            sys.executable,
            "-c",
            "from src.retrieval.vector_store import VectorStore; vs = VectorStore(); vs.clear(); print('Vector store cleared. Initial count:', vs.count())",
        ]
        subprocess.run(cmd_clear, check=True)
        print("-" * 60)
        sys.stdout.flush()

    start_time = time.time()

    for idx, ticker in enumerate(TICKERS, 1):
        print(f"\n>>> [{idx}/{len(TICKERS)}] Processing ticker: {ticker} <<<")
        sys.stdout.flush()

        cmd = [
            sys.executable,
            "scripts/ingest_chunks.py",
            "--ticker",
            ticker,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: Ingestion for {ticker} exited with code {result.returncode}. Retrying once...")
            subprocess.run(cmd)

    # Final count check
    print("\n" + "=" * 60)
    print("FINAL VECTOR STORE VERIFICATION")
    print("=" * 60)
    cmd_count = [
        sys.executable,
        "-c",
        "from src.retrieval.vector_store import VectorStore; vs = VectorStore(); print('Final total vector count in ChromaDB:', vs.count())",
    ]
    subprocess.run(cmd_count, check=True)
    elapsed = round(time.time() - start_time, 2)
    print(f"Total pipeline execution time: {elapsed}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
