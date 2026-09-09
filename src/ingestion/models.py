"""
Data models for parsed SEC 10-K filings.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class FilingSection(BaseModel):
    """
    A structured section or table extracted from an SEC 10-K filing.
    Preserves document structure and rich metadata for downstream RAG retrieval,
    reranking, and source citation.
    """
    model_config = ConfigDict(extra="ignore")

    # Company & Filing Metadata
    company: str = Field(..., description="Company name, e.g. 'Apple Inc.' or 'Apple'")
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'")
    cik: str = Field(..., description="Central Index Key, e.g. '0000320193'")
    filing_type: str = Field(default="10-K", description="Filing type, e.g. '10-K'")
    fiscal_year: int = Field(..., description="Fiscal year, e.g. 2025")
    filing_date: Optional[str] = Field(None, description="Filing submission date, e.g. '2025-10-31'")
    report_date: Optional[str] = Field(None, description="Report period end date, e.g. '2025-09-27'")
    accession_number: Optional[str] = Field(None, description="SEC accession number")

    # Structural Hierarchy
    section: str = Field(
        ...,
        description="Descriptive section name, e.g. 'Consolidated Statements of Operations', 'Item 1A. Risk Factors', 'Note 1 – Summary of Significant Accounting Policies'"
    )
    section_id: str = Field(
        ...,
        description="Machine-readable section key, e.g. 'item_8_operations', 'item_1a', 'item_8_note_1'"
    )
    part: Optional[str] = Field(
        None,
        description="10-K Part if applicable: 'Part I', 'Part II', 'Part III', 'Part IV', 'Cover', etc."
    )

    # Content & Format
    content_type: str = Field(
        default="narrative",
        description="Type of content: 'narrative' or 'table'"
    )
    text: str = Field(..., description="Clean text content or markdown table representation")

    # Page & Location Tracking
    page: Optional[int] = Field(
        None,
        description="Primary page number for citations (matches start page or table page)"
    )
    page_start: Optional[int] = Field(None, description="First page of this section/table")
    page_end: Optional[int] = Field(None, description="Last page of this section/table")

    # Table metadata (when content_type == 'table')
    table_index: Optional[int] = Field(None, description="Index of table within filing or section")

    # Traceability
    source_url: Optional[str] = Field(None, description="Original SEC EDGAR filing URL")
    raw_path: Optional[str] = Field(None, description="Path to raw source filing HTML")
