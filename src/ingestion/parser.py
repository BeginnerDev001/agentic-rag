"""
SEC 10-K filing parser with structure preservation.
Extracts standard 10-K items, breaks down Item 8 financial statements and notes,
tracks physical page numbers, and converts tables to markdown records alongside narrative.
"""

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from src.ingestion.models import FilingSection
from src.preprocessing.cleaner import (
    clean_text,
    html_table_to_markdown,
    strip_xbrl_tags,
)


# Standard 10-K Items with their default display titles and parts
STANDARD_10K_ITEMS = {
    "1": ("Item 1. Business", "Part I"),
    "1A": ("Item 1A. Risk Factors", "Part I"),
    "1B": ("Item 1B. Unresolved Staff Comments", "Part I"),
    "1C": ("Item 1C. Cybersecurity", "Part I"),
    "2": ("Item 2. Properties", "Part I"),
    "3": ("Item 3. Legal Proceedings", "Part I"),
    "4": ("Item 4. Mine Safety Disclosures", "Part I"),
    "5": ("Item 5. Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities", "Part II"),
    "6": ("Item 6. [Reserved]", "Part II"),
    "7": ("Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations", "Part II"),
    "7A": ("Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Part II"),
    "8": ("Item 8. Financial Statements and Supplementary Data", "Part II"),
    "9": ("Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure", "Part II"),
    "9A": ("Item 9A. Controls and Procedures", "Part II"),
    "9B": ("Item 9B. Other Information", "Part II"),
    "9C": ("Item 9C. Disclosure Regarding Foreign Jurisdictions that Prevent Inspections", "Part II"),
    "10": ("Item 10. Directors, Executive Officers and Corporate Governance", "Part III"),
    "11": ("Item 11. Executive Compensation", "Part III"),
    "12": ("Item 12. Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters", "Part III"),
    "13": ("Item 13. Certain Relationships and Related Transactions, and Director Independence", "Part III"),
    "14": ("Item 14. Principal Accountant Fees and Services", "Part III"),
    "15": ("Item 15. Exhibits, Financial Statement Schedules", "Part IV"),
    "16": ("Item 16. Form 10-K Summary", "Part IV"),
}

ORDERED_ITEM_KEYS = [
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A",
    "9B", "9C", "10", "11", "12", "13", "14",
    "15", "16"
]

FINANCIAL_STATEMENT_PATTERNS = [
    (
        re.compile(
            r"^(?:(?:CONSOLIDATED\s+)?(?:STATEMENTS?\s+OF\s+)?COMPREHENSIVE\s+INCOME(?:\s+STATEMENTS?)?|COMPREHENSIVE\s+INCOME\s+STATEMENTS?)(?:\s+(?:for\s+the|years\s+ended|as\s+of).*)?$",
            re.I,
        ),
        "Consolidated Statements of Comprehensive Income",
        "item_8_comprehensive_income",
    ),
    (
        re.compile(
            r"^(?:CONSOLIDATED\s+(?:STATEMENTS?\s+OF\s+)?(?:OPERATIONS|INCOME|EARNINGS)|STATEMENTS?\s+OF\s+(?:OPERATIONS|INCOME|EARNINGS)|(?:CONSOLIDATED\s+)?INCOME\s+STATEMENTS?|STATEMENTS?\s+OF\s+OPERATIONS)(?:\s+(?:for\s+the|years\s+ended|as\s+of).*)?$",
            re.I,
        ),
        "Consolidated Statements of Operations",
        "item_8_operations",
    ),
    (
        re.compile(
            r"^(?:CONSOLIDATED\s+)?BALANCE\s+SHEETS?(?:\s+(?:as\s+of|at).*)?$",
            re.I,
        ),
        "Consolidated Balance Sheets",
        "item_8_balance_sheets",
    ),
    (
        re.compile(
            r"^(?:CONSOLIDATED\s+(?:STATEMENTS?\s+OF\s+)?(?:SHAREHOLDERS|STOCKHOLDERS)[’']?\s+EQUITY|STATEMENTS?\s+OF\s+(?:SHAREHOLDERS|STOCKHOLDERS)[’']?\s+EQUITY)(?:\s+(?:for\s+the|years\s+ended|as\s+of).*)?$",
            re.I,
        ),
        "Consolidated Statements of Shareholders' Equity",
        "item_8_shareholders_equity",
    ),
    (
        re.compile(
            r"^(?:CONSOLIDATED\s+(?:STATEMENTS?\s+OF\s+)?CASH\s+FLOWS?|STATEMENTS?\s+OF\s+CASH\s+FLOWS?|CASH\s+FLOWS?\s+STATEMENTS?)(?:\s+(?:for\s+the|years\s+ended|as\s+of).*)?$",
            re.I,
        ),
        "Consolidated Statements of Cash Flows",
        "item_8_cash_flows",
    ),
    (
        re.compile(
            r"^(?:REPORTS?\s+OF\s+INDEPENDENT\s+REGISTERED\s+PUBLIC\s+ACCOUNTING\s+FIRM)(?:\s+.*)?$",
            re.I,
        ),
        "Report of Independent Registered Public Accounting Firm",
        "item_8_auditor_report",
    ),
]


class SEC10KParser:
    """
    Parser for SEC EDGAR 10-K HTML/iXBRL filings.
    Segments filings into structured sections, Item 8 financial statements, notes,
    and tables with rich metadata.
    """

    def __init__(self):
        pass

    def parse_file(self, file_path: Path | str, metadata: Optional[Dict[str, Any]] = None) -> List[FilingSection]:
        """
        Parse a 10-K filing from a local file path.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        html_content = path.read_text(encoding="utf-8", errors="ignore")
        meta = dict(metadata or {})
        if "raw_path" not in meta:
            meta["raw_path"] = str(path)

        return self.parse_filing(html_content, meta)

    def parse_filing(self, html_content: str, metadata: Dict[str, Any]) -> List[FilingSection]:
        """
        Parse HTML filing into structured FilingSection objects.
        """
        meta = self._normalize_metadata(metadata, html_content)
        pages_raw = self._split_into_pages(html_content)

        sections: List[Dict[str, Any]] = []
        current_item_idx = -1
        in_financials = False

        current_sec: Dict[str, Any] = {
            "title": "Cover Page",
            "section_id": "cover_page",
            "part": "Cover",
            "page_start": 1,
            "page_end": 1,
            "narratives": [],
            "tables": [],  # List of (markdown_table, page_num)
        }

        running_page_num = 1

        for p_idx, p_html in enumerate(pages_raw):
            soup = BeautifulSoup(p_html, "html.parser")
            strip_xbrl_tags(soup)

            page_text = soup.get_text(" ", strip=True)
            page_text_clean = html.unescape(page_text).replace("\xa0", " ")

            detected_p = self._detect_page_number(page_text_clean)
            if detected_p is not None:
                running_page_num = detected_p
            else:
                if p_idx > 1:
                    running_page_num += 1

            p_num = running_page_num

            # Check if this page is part of the Table of Contents index
            if self._is_table_of_contents(p_idx, page_text_clean):
                current_sec["page_end"] = p_num
                continue

            # 1. Scan for headers BEFORE decomposing tables
            for el in soup.find_all(["div", "p", "tr", "h1", "h2", "h3", "b", "strong"]):
                t = el.get_text(" ", strip=True)
                t = html.unescape(t).replace("\xa0", " ")
                t = re.sub(r"\s+", " ", t).strip()

                if not t or len(t) > 160:
                    continue

                # Standard 10-K Item transition
                m_item = re.match(
                    r"^(?:PART\s+[IVX]+[\.\s\:\-]*)?\s*ITEM\s+([0-9]{1,2}[A-Z]?)(?:[\.\:\s\-]+(.*))?$",
                    t,
                    re.I,
                )
                if m_item:
                    item_code = m_item.group(1).upper()
                    if item_code in ORDERED_ITEM_KEYS:
                        idx = ORDERED_ITEM_KEYS.index(item_code)
                        if idx > current_item_idx:
                            current_item_idx = idx
                            if self._has_content(current_sec):
                                sections.append(current_sec)

                            in_financials = (item_code in ["8", "15"])
                            default_title, part = STANDARD_10K_ITEMS[item_code]
                            custom_title = (m_item.group(2) or "").strip()
                            sec_title = f"Item {item_code}. {custom_title}" if custom_title and len(custom_title) > 2 else default_title

                            current_sec = {
                                "title": sec_title,
                                "section_id": f"item_{item_code.lower()}",
                                "part": part,
                                "page_start": p_num,
                                "page_end": p_num,
                                "narratives": [],
                                "tables": [],
                            }
                            continue

                # Financial subheaders (statements or notes)
                if in_financials and len(t) < 130:
                    # Note header (e.g. Note 1 – Summary of Significant Accounting Policies)
                    m_note = re.match(r"^NOTE\s+(\d+)[\s\–\-\:\.]+\s*(.*)$", t, re.I)
                    if m_note:
                        note_num = m_note.group(1)
                        note_title = m_note.group(2).strip()
                        sub_id = f"item_8_note_{note_num}"
                        if sub_id != current_sec["section_id"]:
                            if self._has_content(current_sec):
                                sections.append(current_sec)
                            display_title = f"Note {note_num} – {note_title}" if note_title else f"Note {note_num}"
                            current_sec = {
                                "title": display_title,
                                "section_id": sub_id,
                                "part": "Part II",
                                "page_start": p_num,
                                "page_end": p_num,
                                "narratives": [],
                                "tables": [],
                            }
                            continue

                    # Statement header (Operations, Comprehensive Income, Balance Sheet, Cash Flows, Auditor)
                    found_stmt = False
                    for stmt_regex, stmt_name, stmt_id in FINANCIAL_STATEMENT_PATTERNS:
                        if stmt_regex.match(t):
                            if stmt_id != current_sec["section_id"]:
                                if self._has_content(current_sec):
                                    sections.append(current_sec)
                                current_sec = {
                                    "title": stmt_name,
                                    "section_id": stmt_id,
                                    "part": "Part II",
                                    "page_start": p_num,
                                    "page_end": p_num,
                                    "narratives": [],
                                    "tables": [],
                                }
                                found_stmt = True
                                break
                    if found_stmt:
                        continue

            # 2. Extract tables from page
            page_tables = soup.find_all("table")
            for tbl in page_tables:
                md_tbl = html_table_to_markdown(tbl)
                if md_tbl:
                    current_sec["tables"].append((md_tbl, p_num))
                tbl.decompose()

            # 3. Add cleaned page narrative
            page_narrative = clean_text(soup.get_text("\n", strip=True))
            if page_narrative:
                current_sec["narratives"].append(page_narrative)
            current_sec["page_end"] = p_num

        # Append final section
        if self._has_content(current_sec):
            sections.append(current_sec)

        # Assemble final FilingSection objects
        output_records: List[FilingSection] = []
        table_counter = 1

        for sec in sections:
            sec_title = sec["title"]
            sec_id = sec["section_id"]
            sec_part = sec["part"]
            page_start = sec["page_start"]
            page_end = sec["page_end"]
            combined_text = "\n\n".join(sec["narratives"]).strip()

            # 1. Output narrative record
            if combined_text:
                output_records.append(
                    FilingSection(
                        company=meta["company"],
                        ticker=meta["ticker"],
                        cik=meta["cik"],
                        filing_type=meta.get("form", meta.get("filing_type", "10-K")),
                        fiscal_year=meta["fiscal_year"],
                        filing_date=meta.get("filing_date"),
                        report_date=meta.get("report_date"),
                        accession_number=meta.get("accession_number"),
                        section=sec_title,
                        section_id=sec_id,
                        part=sec_part,
                        content_type="narrative",
                        text=combined_text,
                        page=page_start,
                        page_start=page_start,
                        page_end=page_end,
                        source_url=meta.get("url", meta.get("source_url")),
                        raw_path=meta.get("raw_path"),
                    )
                )

            # 2. Output separate table records
            for tbl_md, tbl_page in sec["tables"]:
                output_records.append(
                    FilingSection(
                        company=meta["company"],
                        ticker=meta["ticker"],
                        cik=meta["cik"],
                        filing_type=meta.get("form", meta.get("filing_type", "10-K")),
                        fiscal_year=meta["fiscal_year"],
                        filing_date=meta.get("filing_date"),
                        report_date=meta.get("report_date"),
                        accession_number=meta.get("accession_number"),
                        section=sec_title,
                        section_id=sec_id,
                        part=sec_part,
                        content_type="table",
                        text=tbl_md,
                        page=tbl_page,
                        page_start=tbl_page,
                        page_end=tbl_page,
                        table_index=table_counter,
                        source_url=meta.get("url", meta.get("source_url")),
                        raw_path=meta.get("raw_path"),
                    )
                )
                table_counter += 1

        return output_records

    def _has_content(self, sec: Dict[str, Any]) -> bool:
        """Check if section has any narrative or tables."""
        return bool(sec["narratives"] or sec["tables"])

    def _normalize_metadata(self, metadata: Dict[str, Any], html_content: str) -> Dict[str, Any]:
        """Ensure required metadata fields are populated."""
        meta = dict(metadata)
        if "ticker" not in meta:
            meta["ticker"] = "UNKNOWN"
        if "company" not in meta:
            meta["company"] = meta["ticker"]
        if "cik" not in meta:
            meta["cik"] = ""

        if "fiscal_year" not in meta or not meta["fiscal_year"]:
            date_val = meta.get("filing_date") or meta.get("report_date") or ""
            if date_val and len(date_val) >= 4 and date_val[:4].isdigit():
                meta["fiscal_year"] = int(date_val[:4])
            else:
                m_yr = re.search(r"fiscal\s+year\s+ended[^\d]{0,30}(\d{4})", html_content, re.I)
                if m_yr:
                    meta["fiscal_year"] = int(m_yr.group(1))
                else:
                    meta["fiscal_year"] = 0
        else:
            meta["fiscal_year"] = int(meta["fiscal_year"])

        return meta

    def _split_into_pages(self, html_content: str) -> List[str]:
        """Split document on page break HR elements or fallback."""
        pages = re.split(r"<hr\b[^>]*>", html_content, flags=re.I)
        if len(pages) <= 1:
            pages = re.split(r"<div[^>]*style=[^>]*page-break-after:[^>]*>", html_content, flags=re.I)
        return pages if len(pages) > 1 else [html_content]

    def _detect_page_number(self, text: str) -> Optional[int]:
        """Detect printed page number from text."""
        m = re.search(r"(?:Form\s+10-K|10-K)\s*\|\s*(\d+)", text, re.I)
        if m:
            return int(m.group(1))

        m = re.search(r"\|\s*(\d{1,4})\s*$", text)
        if m:
            return int(m.group(1))

        m = re.search(r"(?:\n|\s{3,})(\d{1,3})\s*$", text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 500:
                return val

        return None

    def _is_table_of_contents(self, page_idx: int, text: str) -> bool:
        """
        Check if page is part of the introductory Table of Contents.
        The actual Table of Contents page typically lists 8 or more items
        on a single page within the first 5 pages.
        """
        if page_idx > 5:
            return False

        item_matches = re.findall(r"\bItem\s+(?:1A|1B|1C|7A|9A|9B|9C|\d{1,2})\b", text, re.I)
        if len(item_matches) >= 8:
            return True

        return False
