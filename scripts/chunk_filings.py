"""
Chunk all parsed SEC 10-K sections into retrieval-ready chunks.

Reads JSONL section files from data/processed/filings/{TICKER}/ and
writes chunked output to data/processed/chunks/{TICKER}/.

Usage:
    python scripts/chunk_filings.py
    python scripts/chunk_filings.py --ticker AAPL
    python scripts/chunk_filings.py --target-tokens 512 --overlap-tokens 64
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.chunker import SECChunker


SECTIONS_DIR = Path("data/processed/filings")
CHUNKS_DIR = Path("data/processed/chunks")


def main():
    parser = argparse.ArgumentParser(description="Chunk parsed SEC filing sections.")
    parser.add_argument("--ticker", type=str, help="Filter by ticker symbol")
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chunker = SECChunker(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover section files
    section_files = sorted(SECTIONS_DIR.glob("*/*_sections.jsonl"))
    if args.ticker:
        t = args.ticker.upper()
        section_files = [f for f in section_files if f.parent.name == t]

    print(f"Found {len(section_files)} section files to chunk")
    print(f"Config: target={args.target_tokens}, max={args.max_tokens}, overlap={args.overlap_tokens}")
    print()

    total_sections = 0
    total_chunks = 0
    total_tables = 0
    total_narratives = 0
    start = time.time()

    for idx, section_file in enumerate(section_files, 1):
        ticker = section_file.parent.name
        stem = section_file.stem.replace("_sections", "_chunks")

        out_dir = CHUNKS_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.jsonl"

        if out_path.exists() and not args.overwrite:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            print(f"[{idx}/{len(section_files)}] {ticker}/{section_file.name} -> [SKIP] ({existing} chunks)")
            total_chunks += existing
            continue

        # Read sections
        sections = []
        with open(section_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sections.append(json.loads(line))

        total_sections += len(sections)

        # Chunk
        chunks = chunker.chunk_filing(sections)

        # Write
        n_table = 0
        n_narrative = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                if chunk.content_type == "table":
                    n_table += 1
                else:
                    n_narrative += 1
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        total_chunks += len(chunks)
        total_tables += n_table
        total_narratives += n_narrative

        print(
            f"[{idx}/{len(section_files)}] {ticker}/{section_file.name} "
            f"-> {len(chunks)} chunks ({n_narrative} narrative, {n_table} table)"
        )

    elapsed = round(time.time() - start, 2)

    print()
    print("=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Section files processed: {len(section_files)}")
    print(f"Total input sections:    {total_sections}")
    print(f"Total output chunks:     {total_chunks}")
    print(f"  Narrative chunks:      {total_narratives}")
    print(f"  Table chunks:          {total_tables}")
    print(f"Execution time:          {elapsed}s")
    print(f"Output directory:        {CHUNKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
