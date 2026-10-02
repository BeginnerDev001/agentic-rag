"""
Guardrails Module for Agentic Financial RAG.

Provides pre-retrieval input validation and post-generation output abstention
to ensure safe, scoped, and accurate financial analysis.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─── Known Corpus Scope ───────────────────────────────────────────────
VALID_TICKERS = {"AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "V", "WMT"}
VALID_YEARS = set(range(2016, 2027))

# ─── Prompt Injection Patterns ────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"system\s*:\s*",
    r"disregard\s+(everything|all|your)",
    r"pretend\s+you('re|\s+are)\s+",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"roleplay\s+as",
]

# ─── Off-Topic Patterns ──────────────────────────────────────────────
FINANCIAL_KEYWORDS = {
    "revenue", "income", "profit", "loss", "expense", "margin", "earnings",
    "sales", "debt", "equity", "cash", "asset", "liability", "dividend",
    "eps", "share", "stock", "fiscal", "quarter", "annual", "10-k", "10k",
    "sec", "filing", "balance", "sheet", "statement", "ebitda", "capex",
    "operating", "net", "gross", "tax", "rate", "growth", "segment",
    "r&d", "research", "development", "cost", "depreciation", "amortization",
    "goodwill", "impairment", "acquisition", "merger", "risk", "factor",
    "outlook", "guidance", "forecast", "backlog", "inventory", "receivable",
    "payable", "employee", "headcount", "cloud", "datacenter", "gpu",
    "iphone", "azure", "aws", "services", "subscription", "billion",
    "million", "thousand", "percent", "%", "yoy", "qoq",
}

# Company name mapping
COMPANY_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "meta": "META", "facebook": "META",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "visa": "V",
    "walmart": "WMT", "wmt": "WMT",
}


class InputGuardrail:
    """
    Pre-retrieval input validation.
    Checks for empty queries, prompt injection, and off-topic requests.
    """

    def __init__(self):
        self.scope_validator = FinancialScopeValidator()

    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Validate user input before processing.

        Returns:
            (is_valid, message) — if not valid, message explains why.
        """
        # 1. Empty / too short
        cleaned = query.strip()
        if not cleaned or len(cleaned) < 5:
            return False, "Query is too short. Please ask a specific financial question."

        # 2. Prompt injection detection
        query_lower = cleaned.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return False, "Query rejected: potential prompt injection detected."

        # 3. Off-topic detection (must contain at least one financial keyword or company name)
        tokens = set(re.findall(r"\b\w+\b", query_lower))
        has_financial_term = bool(tokens & FINANCIAL_KEYWORDS)
        has_company = any(alias in query_lower for alias in COMPANY_ALIASES)

        if not has_financial_term and not has_company:
            return False, (
                "This system answers questions about SEC 10-K filings for "
                f"{', '.join(sorted(VALID_TICKERS))}. "
                "Please ask a financial question about one of these companies."
            )

        return True, "OK"


class FinancialScopeValidator:
    """
    Validates whether the query targets companies/years within the corpus.
    """

    def extract_scope(self, query: str) -> Dict[str, Any]:
        """Extract ticker and fiscal year from query text."""
        query_lower = query.lower()
        detected_ticker = None
        detected_year = None

        # Detect company/ticker
        for alias, ticker in COMPANY_ALIASES.items():
            if alias in query_lower:
                detected_ticker = ticker
                break

        # Detect fiscal year
        year_matches = re.findall(r"\b(20[1-2]\d)\b", query)
        if year_matches:
            detected_year = int(year_matches[-1])

        return {
            "ticker": detected_ticker,
            "fiscal_year": detected_year,
            "ticker_in_corpus": detected_ticker in VALID_TICKERS if detected_ticker else None,
            "year_in_corpus": detected_year in VALID_YEARS if detected_year else None,
        }

    def validate_scope(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if the query targets an in-corpus ticker and year.

        Returns:
            (is_valid, message, scope_info)
        """
        scope = self.extract_scope(query)

        if scope["ticker"] and not scope["ticker_in_corpus"]:
            return False, f"Ticker '{scope['ticker']}' is not in the corpus.", scope

        if scope["fiscal_year"] and not scope["year_in_corpus"]:
            return False, f"Fiscal year {scope['fiscal_year']} is outside corpus range (2016-2026).", scope

        return True, "OK", scope


class OutputGuardrail:
    """
    Post-generation output validation.
    Detects low-confidence or ungrounded answers and triggers abstention.
    """

    ABSTENTION_RESPONSE = "Insufficient information in the provided SEC filings to generate a verified answer."

    def validate(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        retries: int = 0,
        status: str = "unknown",
    ) -> Tuple[bool, str]:
        """
        Validate generated answer for quality.

        Returns:
            (is_acceptable, final_answer) — may override answer with abstention.
        """
        if not answer or not answer.strip():
            return False, self.ABSTENTION_RESPONSE

        # Already a refusal — pass through
        if "insufficient information" in answer.lower():
            return True, answer

        # If exhausted retries with no success
        if status == "refusal_or_exhausted":
            return True, self.ABSTENTION_RESPONSE

        # Check for minimum citation presence
        has_citation = bool(re.search(r"\[[A-Z0-9_/.\s-]+\]", answer))
        if not has_citation and len(answer) > 100:
            # Long answer with no citation is suspicious
            return True, answer + "\n\n⚠️ *Note: This answer could not be verified with specific filing citations.*"

        return True, answer
