"""
Query Rewriter and Decomposer module for SEC 10-K Agentic RAG.

Transforms raw user queries into formal SEC filing terminology and decomposes
multi-part financial questions into standalone sub-queries.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

REWRITE_SYSTEM_PROMPT = """You are a financial search query optimization expert specializing in SEC 10-K filings.
Your job is to rewrite raw user questions into formal SEC EDGAR accounting and legal search terms.

Rules:
1. Rephrase generic user terms into SEC filing terminology (e.g., "money made" -> "net sales / total revenue", "lawsuits" -> "legal proceedings / Item 3", "risks" -> "Item 1A Risk Factors").
2. If feedback is provided on why previous retrieval failed, address the missing information directly.
3. Keep the output concise and targeted (1 short sentence or search phrase).
4. Output ONLY the rewritten search query. Do NOT include intro or explanation text.
"""

DECOMPOSE_SYSTEM_PROMPT = """You are a financial query decomposition assistant.
Your job is to break down multi-part or comparative financial questions into standalone single-company sub-questions.

Rules:
1. If the question compares multiple companies or multiple metrics/years, output each sub-question on a new line.
2. Ensure each sub-question is self-contained (includes ticker/company name and fiscal year if applicable).
3. If the input is already a single simple question, return it unchanged.
4. Output ONLY the sub-questions (one per line). Do NOT include numbering or intro text.
"""


class QueryRewriter:
    """
    LLM-powered query optimization and decomposition for SEC RAG retrieval.
    """

    def __init__(self, model: str = OPENROUTER_MODEL, api_key: str = OPENROUTER_API_KEY):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to invoke OpenRouter API for query transformations."""
        if not self.api_key:
            # Fallback to returning input if no key configured
            return user_prompt.strip()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }

        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return user_prompt.strip()
        except Exception:
            return user_prompt.strip()

    def rewrite(self, query: str, feedback: Optional[str] = None) -> str:
        """
        Rewrite a user query into formal SEC filing terminology.

        Args:
            query: Raw user input query.
            feedback: Optional feedback string explaining why previous retrieval failed.

        Returns:
            Optimized SEC search query.
        """
        user_prompt = f"Original Query: {query}"
        if feedback:
            user_prompt += f"\nRetrieval Feedback: {feedback}"

        rewritten = self._call_llm(REWRITE_SYSTEM_PROMPT, user_prompt)
        # Strip any quotes if added by LLM
        return rewritten.strip('"\'')

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex or multi-company question into standalone sub-queries.

        Args:
            query: User input question.

        Returns:
            List of standalone sub-queries.
        """
        response = self._call_llm(DECOMPOSE_SYSTEM_PROMPT, query)
        lines = [line.strip("- *1234567890.").strip() for line in response.splitlines() if line.strip()]
        return lines if lines else [query]
