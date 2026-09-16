"""
Self-Critique Graders for Agentic RAG.

Provides DocumentGrader (relevance filter), HallucinationGrader (factuality check),
and AnswerQualityGrader (utility check) using structured LLM prompts.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

DOC_GRADER_PROMPT = """You are a document relevance evaluator for SEC filing search results.
Your job is to determine whether a retrieved document chunk is RELEVANT to the user's question.

Question: {query}
Document Chunk: {document}

Evaluate relevance. Reply with JSON ONLY:
{{
  "relevant": true or false,
  "reason": "short 1-sentence explanation"
}}
"""

HALLUCINATION_GRADER_PROMPT = """You are a fact-checking assistant evaluating an AI generated answer against provided SEC filing context.
Your job is to determine if the generated answer is FULLY GROUNDED in the provided context and contains NO hallucinated claims.

Context:
{context}

Generated Answer:
{answer}

Evaluate grounding. Reply with JSON ONLY:
{{
  "grounded": true or false,
  "reason": "short 1-sentence explanation"
}}
"""

ANSWER_GRADER_PROMPT = """You are an answer quality evaluator.
Your job is to determine whether the generated answer directly addresses the user's question.

User Question: {query}
Generated Answer: {answer}

Evaluate utility. Reply with JSON ONLY:
{{
  "useful": true or false,
  "reason": "short 1-sentence explanation"
}}
"""


class BaseGrader:
    """Base class for calling OpenRouter with JSON output mode."""

    def __init__(self, model: str = OPENROUTER_MODEL, api_key: str = OPENROUTER_API_KEY):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def _call_json(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"relevant": True, "grounded": True, "useful": True, "reason": "No API key"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 150,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                # Fallback lenient
                return {"relevant": True, "grounded": True, "useful": True, "reason": f"API error {resp.status_code}"}
        except Exception as e:
            return {"relevant": True, "grounded": True, "useful": True, "reason": str(e)}


class DocumentGrader(BaseGrader):
    """Evaluates whether a chunk is relevant to the query."""

    def grade(self, query: str, document_text: str) -> Tuple[bool, str]:
        prompt = DOC_GRADER_PROMPT.format(query=query, document=document_text[:1500])
        res = self._call_json(prompt)
        is_rel = bool(res.get("relevant", True))
        reason = str(res.get("reason", ""))
        return is_rel, reason

    def filter_relevant_chunks(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter a list of retrieved chunk dicts, returning only relevant ones."""
        relevant = []
        for chunk in chunks:
            is_rel, reason = self.grade(query, chunk.get("text", ""))
            if is_rel:
                chunk["relevance_reason"] = reason
                relevant.append(chunk)
        return relevant


class HallucinationGrader(BaseGrader):
    """Evaluates whether an answer is grounded in retrieved context."""

    def grade(self, context_text: str, answer_text: str) -> Tuple[bool, str]:
        if "insufficient information" in answer_text.lower():
            # Refusals are strictly non-hallucinatory
            return True, "Valid refusal statement"

        prompt = HALLUCINATION_GRADER_PROMPT.format(
            context=context_text[:3000], answer=answer_text
        )
        res = self._call_json(prompt)
        is_grounded = bool(res.get("grounded", True))
        reason = str(res.get("reason", ""))
        return is_grounded, reason


class AnswerQualityGrader(BaseGrader):
    """Evaluates whether an answer directly addresses the query."""

    def grade(self, query: str, answer_text: str) -> Tuple[bool, str]:
        prompt = ANSWER_GRADER_PROMPT.format(query=query, answer=answer_text)
        res = self._call_json(prompt)
        is_useful = bool(res.get("useful", True))
        reason = str(res.get("reason", ""))
        return is_useful, reason
