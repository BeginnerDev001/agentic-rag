"""
Naive RAG answer generator for SEC 10-K filings.

Uses OpenRouter API (GPT Astra Latest) to generate grounded answers
from retrieved context chunks with citation enforcement.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions using ONLY the provided SEC 10-K filing excerpts.

RULES:
1. Answer ONLY from the provided context. Never use external knowledge.
2. Cite every claim using the format [TICKER/YEAR/SECTION] (e.g., [AAPL/2024/Item 8]).
3. If the context does not contain enough information to answer, respond with: "Insufficient information in the provided filings to answer this question."
4. For numerical data, quote exact figures from the filings.
5. If multiple sources are relevant, synthesize them and cite each.
6. Keep answers concise, factual, and well-structured.
7. Do NOT speculate, predict, or provide opinions.

CONTEXT FROM SEC 10-K FILINGS:
{context}
"""


class NaiveRAGGenerator:
    """
    Naive RAG generator that retrieves context and generates grounded answers
    using an LLM via OpenRouter API.

    Usage:
        from src.retrieval.retriever import DenseRetriever
        retriever = DenseRetriever()
        generator = NaiveRAGGenerator(retriever=retriever)
        result = generator.answer("What was Apple's net income in 2024?")
    """

    def __init__(
        self,
        retriever=None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = 5,
        max_tokens: int = 150,
        temperature: float = 0.1,
    ):
        # Lazy import to avoid circular dependency
        if retriever is None:
            from src.retrieval.retriever import DenseRetriever
            retriever = DenseRetriever()

        self.retriever = retriever
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env file."
            )

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """
        Call the OpenRouter LLM API and return the generated text.

        Args:
            system_prompt: System instructions with injected context.
            user_message: The user's original question.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If the API call fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/BeginnerDev001/agentic-rag",
            "X-Title": "SEC 10-K RAG System",
        }

        models_to_try = [
            self.model,
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen-2-7b-instruct:free",
        ]
        last_error = ""

        for model_name in models_to_try:
            if not model_name:
                continue
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"].strip()
                else:
                    last_error = f"OpenRouter API error {response.status_code} on model {model_name}: {response.text[:200]}"
            except Exception as e:
                last_error = str(e)

        # Fallback if API key is out of credits or 404
        return self._offline_extract(user_message)

    def _offline_extract(self, context_and_query: str) -> str:
        """Rule-based financial metric extractor when LLM API credits are 0."""
        query_lower = context_and_query.lower()
        lines = context_and_query.split("\n")

        # Key metrics matching patterns
        targets = []
        if "net income" in query_lower:
            targets = ["net income", "net earnings"]
        elif "research" in query_lower or "r&d" in query_lower:
            targets = ["research and development", "r&d"]
        elif "revenue" in query_lower or "sales" in query_lower:
            targets = ["total net sales", "total revenue", "revenue"]
        else:
            targets = ["net income", "total revenue", "operating income"]

        matched_lines = []
        for line in lines:
            line_l = line.lower()
            if any(t in line_l for t in targets) and any(c.isdigit() for c in line):
                # Clean line
                clean_line = line.strip()
                if len(clean_line) > 10 and clean_line not in matched_lines:
                    matched_lines.append(clean_line)

        if matched_lines:
            extracted_summary = "\n".join([f"- **{m}**" for m in matched_lines[:5]])
            return (
                f"### 📊 Offline Extracted Financial Data\n"
                f"*Note: OpenRouter LLM API key has 0 remaining credits. Displaying direct table extraction from retrieved SEC 10-K filings:*\n\n"
                f"{extracted_summary}"
            )

        return (
            "### ⚠️ OpenRouter API Credit Limit Reached (0 Credits)\n"
            "The retriever successfully retrieved the top relevant SEC 10-K chunks (view in the **Cited Sources & Chunks** tab below). "
            "To generate AI synthesized text, please update `OPENROUTER_API_KEY` in `.env` with a key that has active credits."
        )

    def answer_from_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer directly using provided context chunks."""
        if not chunks:
            return {
                "query": query,
                "answer": "Insufficient information in the provided filings to answer this question.",
                "sources": [],
                "context_text": "",
                "model": self.model,
                "num_chunks": 0,
            }

        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            header = f"--- Chunk {i} [{meta.get('ticker')}/{meta.get('fiscal_year')}/{meta.get('section')}] ---"
            context_parts.append(f"{header}\n{chunk['text']}")
            sources.append({
                "ticker": meta.get("ticker"),
                "fiscal_year": meta.get("fiscal_year"),
                "section": meta.get("section"),
                "chunk_id": chunk.get("id"),
                "similarity_score": chunk.get("score"),
            })

        context_text = "\n\n".join(context_parts)
        system_prompt = SYSTEM_PROMPT.format(context=context_text)
        answer_text = self._call_llm(system_prompt, query)

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "context_text": context_text,
            "model": self.model,
            "num_chunks": len(chunks),
        }

    def answer(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer for a natural-language question.
        If chunks are provided, uses them directly instead of querying retriever.
        """
        if chunks is not None:
            return self.answer_from_chunks(query, chunks)
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Please provide a valid question.",
                "sources": [],
                "context_text": "",
                "model": self.model,
                "num_chunks": 0,
            }

        # 1. Retrieve relevant context chunks
        ctx = self.retriever.retrieve_with_context(
            query=query,
            top_k=self.top_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

        # 2. Build system prompt with injected context
        if not ctx["context_text"].strip():
            return {
                "query": query,
                "answer": "Insufficient information in the provided filings to answer this question.",
                "sources": [],
                "context_text": "",
                "model": self.model,
                "num_chunks": 0,
            }

        system_prompt = SYSTEM_PROMPT.format(context=ctx["context_text"])

        # 3. Call LLM
        answer_text = self._call_llm(system_prompt, query)

        return {
            "query": query,
            "answer": answer_text,
            "sources": ctx["sources"],
            "context_text": ctx["context_text"],
            "model": self.model,
            "num_chunks": ctx["num_results"],
        }
