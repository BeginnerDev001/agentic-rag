import json
import time
from pathlib import Path

import requests
import yaml


# --------------------------------------------------
# Configuration
# --------------------------------------------------

YEARS = 10

CONFIG_PATH = Path("config/companies.yaml")
RAW_DIR = Path("data/raw/sec")
MANIFEST_PATH = Path("data/manifests/filings.jsonl")

# SEC requires a descriptive User-Agent
HEADERS = {
    "User-Agent": "agentic_rag saharshjeetsingh@gmail.com"
}

SEC_SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{cik}.json"
)

SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_companies():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config["companies"]


def get_company_cik(ticker):
    """
    Get SEC CIK from the SEC company ticker mapping.
    """

    url = "https://www.sec.gov/files/company_tickers.json"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    companies = response.json()

    ticker = ticker.upper()

    for company in companies.values():
        if company["ticker"] == ticker:
            return str(company["cik_str"]).zfill(10)

    raise ValueError(f"Could not find CIK for {ticker}")


def get_10k_filings(cik):
    """
    Get 10-K filings for a company.
    """

    url = SEC_SUBMISSIONS_URL.format(cik=cik)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    recent = data["filings"]["recent"]

    filings = []

    for i, form in enumerate(recent["form"]):

        if form != "10-K":
            continue

        filing = {
            "accession_number": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "form": form,
            "primary_document": recent["primaryDocument"][i],
        }

        filings.append(filing)

    return filings


def build_filing_url(cik, accession_number, primary_document):
    """
    Construct SEC filing URL.
    """

    cik_without_zeroes = str(int(cik))

    accession_no_dash = accession_number.replace("-", "")

    return (
        f"{SEC_ARCHIVE_URL}"
        f"{cik_without_zeroes}/"
        f"{accession_no_dash}/"
        f"{primary_document}"
    )


def download_file(url, output_path):
    """
    Download a filing.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    companies = load_companies()

    all_records = []

    for company in companies:

        ticker = company["ticker"]
        name = company["name"]

        print(f"\nProcessing {ticker}...")

        # ------------------------------------------
        # Find CIK
        # ------------------------------------------

        cik = get_company_cik(ticker)

        print(f"CIK: {cik}")

        # ------------------------------------------
        # Get 10-K filings
        # ------------------------------------------

        filings = get_10k_filings(cik)

        # Take only last N filings
        filings = filings[:YEARS]

        company_dir = RAW_DIR / ticker
        company_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ------------------------------------------
        # Download
        # ------------------------------------------

        for filing in filings:

            accession = filing["accession_number"]
            filing_date = filing["filing_date"]
            primary_document = filing["primary_document"]

            year = filing_date[:4]

            output_path = (
                company_dir /
                f"{year}_{accession}.html"
            )

            url = build_filing_url(
                cik,
                accession,
                primary_document
            )

            print(
                f"  {year} 10-K -> "
                f"{output_path}"
            )

            # Don't download again if already present
            if output_path.exists():

                print("    Already downloaded.")

                status = "already_exists"

            else:

                try:

                    download_file(
                        url,
                        output_path
                    )

                    print("    Downloaded.")

                    status = "downloaded"

                    # Be polite to SEC
                    time.sleep(0.2)

                except Exception as e:

                    print(
                        f"    ERROR: {e}"
                    )

                    status = "failed"

            # --------------------------------------
            # Manifest record
            # --------------------------------------

            record = {
                "ticker": ticker,
                "company": name,
                "cik": cik,
                "form": filing["form"],
                "filing_date": filing_date,
                "report_date": filing["report_date"],
                "accession_number": accession,
                "primary_document": primary_document,
                "url": url,
                "raw_path": str(output_path),
                "download_status": status
            }

            all_records.append(record)

        # Avoid hammering SEC
        time.sleep(0.5)

    # ----------------------------------------------
    # Write manifest
    # ----------------------------------------------

    with open(
        MANIFEST_PATH,
        "w"
    ) as f:

        for record in all_records:

            f.write(
                json.dumps(record)
                + "\n"
            )

    print("\n--------------------------------")
    print("Done!")
    print(f"Filings: {len(all_records)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Raw data: {RAW_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
