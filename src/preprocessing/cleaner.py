"""
HTML cleaning and normalization utilities for SEC 10-K filings.
Handles iXBRL element stripping, entity decoding, whitespace normalization,
and structured table-to-markdown conversion.
"""

import html
import re
from typing import List, Optional
from bs4 import BeautifulSoup, Tag


# Common currency symbols
CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from SEC filings.
    - Decodes HTML entities (e.g. &nbsp;, &#8217;)
    - Normalizes special whitespace and dashes
    - Normalizes typographic quotes
    - Collapses consecutive whitespace while preserving paragraph breaks
    """
    if not text:
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Replace special whitespace characters with standard ASCII space
    text = re.sub(r"[\xa0\u2000-\u200b\u202f\u205f\u3000\ufeff]", " ", text)

    # Normalize typographic quotes and apostrophes
    text = re.sub(r"[\u2018\u2019\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201f]", '"', text)

    # Normalize typographic dashes
    text = re.sub(r"[\u2013\u2014\u2015]", " - ", text)

    # Remove non-printable control characters (keep \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize horizontal whitespace (spaces, tabs)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    # Collapse multiple consecutive empty lines to a maximum of 2
    collapsed = "\n".join(cleaned_lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)

    return collapsed.strip()


def strip_xbrl_tags(soup: BeautifulSoup) -> None:
    """
    Clean BeautifulSoup DOM by decomposing invisible elements and unwrapping
    inline XBRL tags to retain their human-readable content.
    """
    # 1. Remove invisible / metadata elements completely
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    for hidden in soup.find_all(attrs={"style": lambda s: s and "display:none" in s.lower().replace(" ", "")}):
        hidden.decompose()

    for ix_meta in soup.find_all(lambda t: t.name and t.name.startswith("ix:") and t.name in [
        "ix:header", "ix:hidden", "ix:references", "ix:resources"
    ]):
        ix_meta.decompose()

    # 2. Unwrap inline XBRL elements so their inner text is preserved
    for ix_tag in soup.find_all(lambda t: t.name and (t.name.startswith("ix:") or t.name.startswith("xbrli:"))):
        ix_tag.unwrap()


def clean_cell_text(cell: Tag) -> str:
    """
    Extract and clean text from a table cell (td/th).
    """
    txt = cell.get_text(separator=" ", strip=True)
    txt = html.unescape(txt)
    txt = re.sub(r"[\xa0\u2000-\u200b\u202f\u205f\u3000\ufeff]", " ", txt)
    txt = re.sub(r"[\u2018\u2019\u201b]", "'", txt)
    txt = re.sub(r"[\u201c\u201d\u201f]", '"', txt)
    txt = re.sub(r"[\u2013\u2014]", "-", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    # Normalize currency symbol followed by space and digit (e.g. $ 416,161 -> $416,161)
    txt = re.sub(r"([$€£¥])\s+(\d)", r"\1\2", txt)
    # Avoid breaking markdown table formatting by replacing pipes
    txt = txt.replace("|", "/")
    return txt


def html_table_to_markdown(table_elem: Tag) -> Optional[str]:
    """
    Convert an HTML <table> element into a clean markdown table.
    Filters decorative/empty tables, prunes spacer columns, and merges split
    currency/percentage tokens with values.
    Returns None if table has no meaningful tabular content.
    """
    rows = table_elem.find_all("tr")
    if not rows:
        return None

    raw_grid: List[List[str]] = []
    for row in rows:
        cells = [clean_cell_text(c) for c in row.find_all(["th", "td"])]
        if any(c for c in cells):
            raw_grid.append(cells)

    if not raw_grid:
        return None

    # Determine maximum number of columns
    max_cols = max(len(r) for r in raw_grid)
    if max_cols < 2 and len(raw_grid) < 2:
        # Single cell or empty table
        return None

    # Pad all rows to max_cols
    padded = [r + [""] * (max_cols - len(r)) for r in raw_grid]

    # Count non-empty entries per column to identify and drop spacer columns
    non_empty_counts = [sum(1 for r in padded if r[c].strip()) for c in range(max_cols)]
    
    # Require at least one non-empty value
    keep_cols = [c for c in range(max_cols) if non_empty_counts[c] > 0]
    # If the table has several rows, prune columns that contain only 1 rare stray artifact
    if len(raw_grid) >= 4:
        keep_cols = [c for c in keep_cols if non_empty_counts[c] >= 2 or max(non_empty_counts) <= 2]

    if len(keep_cols) < 2 and len(raw_grid) < 2:
        return None

    if not keep_cols:
        return None

    filtered = [[r[c] for c in keep_cols] for r in padded]

    # Merge isolated currency symbols with numbers and percentages with values
    merged_grid = []
    for r in filtered:
        new_row = []
        skip_next = False
        for i in range(len(r)):
            if skip_next:
                skip_next = False
                continue
            val = r[i]
            # Merge standalone currency symbol with following number
            if val in CURRENCY_SYMBOLS and i + 1 < len(r) and r[i + 1]:
                new_row.append(val + r[i + 1])
                skip_next = True
            # Merge following percentage symbol with preceding number
            elif i + 1 < len(r) and r[i + 1] == "%":
                new_row.append((val + " %").strip())
                skip_next = True
            else:
                new_row.append(val)
        merged_grid.append(new_row)

    if not merged_grid:
        return None

    final_max_cols = max(len(r) for r in merged_grid)
    if final_max_cols == 0:
        return None

    final_grid = [r + [""] * (final_max_cols - len(r)) for r in merged_grid]

    # Check total non-empty cell count - must have at least some substantial data
    total_non_empty = sum(1 for r in final_grid for c in r if c.strip())
    if total_non_empty < 3:
        return None

    # Build markdown table lines
    header = final_grid[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * final_max_cols) + " |")
    for r in final_grid[1:]:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)
