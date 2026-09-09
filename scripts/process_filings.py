"""
Process SEC 10-K filings into structured JSONL sections.
Reads from data/manifests/filings.jsonl, parses each filing HTML,
and outputs structured JSONL files to data/processed/filings/{ticker}/.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parser import SEC10KParser


MANIFEST_PATH = Path("data/manifests/filings.jsonl")
PARSED_MANIFEST_PATH = Path("data/manifests/parsed_filings.jsonl")
PROCESSED_DIR = Path("data/processed/filings")


def load_filing_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load downloaded filings from manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_single_filing(
    record: Dict[str, Any],
    parser: SEC10KParser,
    output_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Parse a single filing and save structured sections to JSONL.
    """
    ticker = record["ticker"]
    company = record.get("company", ticker)
    accession = record["accession_number"]
    filing_date = record.get("filing_date", "")
    raw_path_str = record.get("raw_path", "")
    raw_path = Path(raw_path_str)

    # Determine fiscal year
    year = filing_date[:4] if filing_date else "unknown"
    if record.get("report_date"):
        year = record["report_date"][:4]

    company_dir = output_dir / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{year}_{accession}_sections.jsonl"
    output_path = company_dir / output_filename

    # Skip if already processed and overwrite is False
    if output_path.exists() and not overwrite:
        # Read existing line count for statistics
        with open(output_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        return {
            "ticker": ticker,
            "company": company,
            "fiscal_year": int(year) if year.isdigit() else 0,
            "accession_number": accession,
            "output_path": str(output_path),
            "total_sections": total_lines,
            "narrative_sections": 0,
            "table_sections": 0,
            "duration_seconds": 0.0,
            "status": "already_exists",
        }

    if not raw_path.exists():
        return {
            "ticker": ticker,
            "company": company,
            "fiscal_year": int(year) if year.isdigit() else 0,
            "accession_number": accession,
            "output_path": str(output_path),
            "total_sections": 0,
            "duration_seconds": 0.0,
            "status": "raw_file_not_found",
            "error": f"Raw filing not found at {raw_path}",
        }

    start_time = time.time()
    try:
        # Provide manifest metadata to parser
        meta = {
            "company": company,
            "ticker": ticker,
            "cik": record.get("cik", ""),
            "filing_type": record.get("form", "10-K"),
            "filing_date": filing_date,
            "report_date": record.get("report_date"),
            "accession_number": accession,
            "fiscal_year": int(year) if year.isdigit() else 0,
            "url": record.get("url"),
            "raw_path": str(raw_path),
        }

        sections = parser.parse_file(raw_path, meta)
        duration = round(time.time() - start_time, 2)

        # Write output JSONL
        narrative_count = 0
        table_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sec in sections:
                if sec.content_type == "narrative":
                    narrative_count += 1
                elif sec.content_type == "table":
                    table_count += 1
                f.write(sec.model_dump_json() + "\n")

        return {
            "ticker": ticker,
            "company": company,
            "fiscal_year": int(year) if year.isdigit() else 0,
            "accession_number": accession,
            "output_path": str(output_path),
            "total_sections": len(sections),
            "narrative_sections": narrative_count,
            "table_sections": table_count,
            "duration_seconds": duration,
            "status": "success",
        }

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        return {
            "ticker": ticker,
            "company": company,
            "fiscal_year": int(year) if year.isdigit() else 0,
            "accession_number": accession,
            "output_path": str(output_path),
            "total_sections": 0,
            "duration_seconds": duration,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser_cli = argparse.ArgumentParser(description="Parse SEC 10-K filings into structured JSONL sections.")
    parser_cli.add_argument("--ticker", type=str, help="Filter by ticker symbol (e.g. AAPL)")
    parser_cli.add_argument("--limit", type=int, help="Limit number of filings to process")
    parser_cli.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to input manifest")
    parser_cli.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Directory to save JSONL output")
    parser_cli.add_argument("--overwrite", action="store_true", help="Overwrite existing parsed JSONL files")

    args = parser_cli.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    PARSED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.manifest}...")
    filings = load_filing_manifest(args.manifest)

    # Filter by ticker if specified
    if args.ticker:
        target_ticker = args.ticker.upper()
        filings = [f for f in filings if f.get("ticker", "").upper() == target_ticker]

    # Apply limit if specified
    if args.limit and args.limit > 0:
        filings = filings[:args.limit]

    print(f"Total filings to process: {len(filings)}")

    parser = SEC10KParser()
    parsed_records = []

    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_sections_count = 0
    total_tables_count = 0

    overall_start = time.time()

    for idx, filing in enumerate(filings, 1):
        ticker = filing.get("ticker", "UNKNOWN")
        accession = filing.get("accession_number", "")
        print(f"[{idx}/{len(filings)}] Processing {ticker} ({accession})...", end=" ", flush=True)

        res = process_single_filing(filing, parser, args.output_dir, overwrite=args.overwrite)
        parsed_records.append(res)

        status = res["status"]
        if status == "success":
            success_count += 1
            total_sections_count += res["total_sections"]
            total_tables_count += res.get("table_sections", 0)
            print(f"✓ Done: {res['total_sections']} sections ({res['narrative_sections']} narrative, {res['table_sections']} tables) in {res['duration_seconds']}s")
        elif status == "already_exists":
            skipped_count += 1
            total_sections_count += res["total_sections"]
            print(f"→ Already exists ({res['total_sections']} records)")
        else:
            failed_count += 1
            err_msg = res.get("error", "unknown error")
            print(f"✗ Failed: {err_msg}")

    # Write parsed manifest
    with open(PARSED_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in parsed_records:
            f.write(json.dumps(r) + "\n")

    total_time = round(time.time() - overall_start, 2)
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)
    print(f"Total filings evaluated: {len(filings)}")
    print(f"  Successfully parsed:  {success_count}")
    print(f"  Already existed:      {skipped_count}")
    print(f"  Failed:               {failed_count}")
    print(f"Total structured sections created: {total_sections_count}")
    print(f"Total tables extracted:            {total_tables_count}")
    print(f"Total execution time:              {total_time}s")
    print(f"Output directory:                  {args.output_dir}")
    print(f"Parsed manifest:                   {PARSED_MANIFEST_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
