"""
Evaluation metrics for SEC 10-K RAG pipeline.

Defines retrieval metrics (Hit@K, MRR) and generation metrics (Citation Rate, Refusal Accuracy, Answer Match).
"""

import re
from typing import Any, Dict, List, Optional


def compute_retrieval_hit(
    retrieved_chunks: List[Dict[str, Any]],
    expected_company: str,
    expected_section: Optional[str] = None,
) -> float:
    """
    Check if at least one retrieved chunk matches the expected company.

    Returns:
        1.0 if a chunk matching the expected company is in top-k, 0.0 otherwise.
    """
    if not retrieved_chunks:
        return 0.0

    expected_company = expected_company.upper()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        chunk_company = meta.get("ticker", "").upper()

        if chunk_company == expected_company:
            return 1.0

    return 0.0


def compute_mrr(
    retrieved_chunks: List[Dict[str, Any]],
    expected_company: str,
    expected_section: Optional[str] = None,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for the top-k retrieved chunks.

    Returns:
        1/rank of first matching chunk (1.0, 0.5, 0.33, etc.), or 0.0 if no match.
    """
    if not retrieved_chunks:
        return 0.0

    expected_company = expected_company.upper()
    for rank, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        chunk_company = meta.get("ticker", "").upper()

        if chunk_company == expected_company:
            return 1.0 / rank

    return 0.0


def check_has_citation(answer_text: str) -> bool:
    """
    Check if the generated answer contains at least one source citation like [AAPL/2024/Item 8].
    """
    citation_pattern = r"\[[A-Z0-9_-]+/\d{4}/[^\]]+\]"
    return bool(re.search(citation_pattern, answer_text))


def check_is_refusal(answer_text: str) -> bool:
    """
    Check if the model generated an explicit refusal / insufficient information response.
    """
    refusal_phrases = [
        "insufficient information",
        "does not contain enough information",
        "no information provided",
        "not mentioned in the provided filings",
        "cannot answer",
    ]
    text_lower = answer_text.lower()
    return any(phrase in text_lower for phrase in refusal_phrases)


def compute_numeric_overlap(expected: str, generated: str) -> float:
    """
    Extract key numbers/percentages from expected answer and check if they appear in generated.

    Returns:
        Fraction of expected numbers found in generated text (0.0 to 1.0).
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", expected)
    if not numbers:
        return 1.0  # No numbers to verify

    found_count = sum(1 for num in numbers if num in generated)
    return found_count / len(numbers)
