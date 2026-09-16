"""
Agentic RAG Orchestrator with Self-Correction Loop.

Coordinates query re-writing, vector retrieval, document relevance grading,
LLM answer synthesis, and hallucination self-critique.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.grader import AnswerQualityGrader, DocumentGrader, HallucinationGrader
from src.agent.rewriter import QueryRewriter
from src.generation.answer import NaiveRAGGenerator
from src.retrieval.retriever import DenseRetriever


class AgenticRAG:
    """
    Autonomous SEC 10-K Agentic RAG system with self-correcting retrieval & generation loop.

    Workflow:
      1. Query Decompose / Rewrite -> Formulate optimized search term.
      2. Retrieve -> DenseRetriever search against ChromaDB.
      3. Grade Documents -> Filter out irrelevant boilerplate.
      4. Self-Correct Loop -> If 0 relevant docs, rewrite query with feedback and retry.
      5. Answer Generate -> LLM synthesis with grounded citations.
      6. Grade Hallucination -> Verify answer factuality against context.
    """

    def __init__(
        self,
        retriever: Optional[DenseRetriever] = None,
        generator: Optional[NaiveRAGGenerator] = None,
        max_retries: int = 2,
    ):
        self.retriever = retriever or DenseRetriever()
        self.generator = generator or NaiveRAGGenerator(retriever=self.retriever)
        self.rewriter = QueryRewriter()
        self.doc_grader = DocumentGrader()
        self.hallucination_grader = HallucinationGrader()
        self.answer_grader = AnswerQualityGrader()
        self.max_retries = max_retries

    def ask(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute self-correcting agent pipeline for a financial query.

        Returns:
            Dict containing final answer, cited sources, execution trace, and retry count.
        """
        trace: List[Dict[str, Any]] = []

        # 1. Decompose query if multi-company/multi-part
        sub_queries = self.rewriter.decompose(query)
        if len(sub_queries) > 1:
            return self._handle_multi_query(query, sub_queries, ticker, fiscal_year, section)

        current_query = query
        retries = 0
        final_answer = ""
        final_sources: List[Dict[str, Any]] = []
        status = "unknown"

        while retries <= self.max_retries:
            attempt_info: Dict[str, Any] = {"attempt": retries + 1}

            # A. Rewrite Query
            rewritten_query = self.rewriter.rewrite(
                current_query,
                feedback=f"Attempt {retries}: need targeted SEC facts" if retries > 0 else None,
            )
            attempt_info["rewritten_query"] = rewritten_query

            # B. Retrieve Chunks
            retrieved_chunks = self.retriever.retrieve(
                query=rewritten_query,
                top_k=5,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
            )
            attempt_info["num_retrieved"] = len(retrieved_chunks)

            # C. Grade Documents
            relevant_chunks = self.doc_grader.filter_relevant_chunks(current_query, retrieved_chunks)
            attempt_info["num_relevant"] = len(relevant_chunks)

            if not relevant_chunks:
                # Retry if no relevant docs found
                attempt_info["outcome"] = "no_relevant_docs_retry"
                trace.append(attempt_info)
                retries += 1
                current_query = f"Provide relevant SEC filing data for: {query}"
                continue

            # D. Generate Answer
            gen_res = self.generator.answer(
                query=current_query,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                chunks=relevant_chunks,
            )
            answer_text = gen_res["answer"]
            sources = gen_res["sources"]

            # E. Hallucination Check
            context_text = "\n\n".join(c["text"] for c in relevant_chunks)
            is_grounded, ground_reason = self.hallucination_grader.grade(context_text, answer_text)

            attempt_info["is_grounded"] = is_grounded
            attempt_info["ground_reason"] = ground_reason

            if is_grounded:
                # Success!
                attempt_info["outcome"] = "success"
                trace.append(attempt_info)
                final_answer = answer_text
                final_sources = sources
                status = "success"
                break
            else:
                # Retry if hallucination detected
                attempt_info["outcome"] = "hallucination_retry"
                trace.append(attempt_info)
                retries += 1
                current_query = f"Stick STRICTLY to facts. Question: {query}"

        if not final_answer:
            # Fallback
            final_answer = "Insufficient information in the provided SEC filings to generate a verified answer."
            status = "refusal_or_exhausted"

        return {
            "original_query": query,
            "answer": final_answer,
            "sources": final_sources,
            "retries": retries,
            "status": status,
            "execution_trace": trace,
        }

    def _handle_multi_query(
        self,
        original_query: str,
        sub_queries: List[str],
        ticker: Optional[str],
        fiscal_year: Optional[int],
        section: Optional[str],
    ) -> Dict[str, Any]:
        """Process decomposed sub-queries and combine results."""
        answers = []
        all_sources = []
        traces = []

        for sub_q in sub_queries:
            sub_res = self.ask(sub_q, ticker=ticker, fiscal_year=fiscal_year, section=section)
            answers.append(sub_res["answer"])
            all_sources.extend(sub_res["sources"])
            traces.append(sub_res["execution_trace"])

        combined_answer = "\n\n".join(answers)
        return {
            "original_query": original_query,
            "answer": combined_answer,
            "sources": all_sources,
            "retries": 0,
            "status": "decomposed_success",
            "sub_queries": sub_queries,
            "execution_trace": traces,
        }
