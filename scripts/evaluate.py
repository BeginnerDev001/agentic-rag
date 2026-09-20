"""
CLI script to evaluate the SEC 10-K RAG system across retrieval modes.

Modes:
  dense            — DenseRetriever only (Baseline v0)
  bm25             — BM25Retriever only (sparse)
  hybrid           — DenseRetriever + BM25 fused via RRF
  hybrid_reranked  — Hybrid + CrossEncoder reranker (highest precision)

Usage:
  python scripts/evaluate.py --mode dense --sample 20
  python scripts/evaluate.py --mode hybrid_reranked --sample 20
  python scripts/evaluate.py --mode dense --full
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
torch.set_num_threads(1)


VALID_MODES = ["dense", "bm25", "hybrid", "hybrid_reranked"]


def build_retriever(mode: str):
    """Instantiate the correct retriever for the given mode."""
    if mode == "dense":
        from src.retrieval.retriever import DenseRetriever
        print("[Mode] Dense vector retrieval (Baseline v0)")
        return DenseRetriever()

    elif mode == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        print("[Mode] BM25 sparse retrieval")
        return BM25Retriever()

    elif mode == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        print("[Mode] Hybrid Dense+BM25 with RRF fusion")
        return HybridRetriever()

    elif mode == "hybrid_reranked":
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker

        print("[Mode] Hybrid Dense+BM25 with RRF + CrossEncoder reranker")

        base_retriever = HybridRetriever()
        reranker = CrossEncoderReranker()

        # Wrap into an adapter so evaluator can call .retrieve() uniformly
        class RerankedRetriever:
            def __init__(self, retriever, reranker, fetch_k=20):
                self._retriever = retriever
                self._reranker = reranker
                self._fetch_k = fetch_k

            @property
            def corpus_size(self):
                return self._retriever.corpus_size

            def retrieve(self, query, top_k=5, ticker=None, fiscal_year=None, section=None):
                return self._reranker.retrieve_and_rerank(
                    query=query,
                    retriever=self._retriever,
                    top_k=top_k,
                    fetch_k=self._fetch_k,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    section=section,
                )

            def retrieve_with_context(self, query, top_k=5, ticker=None, fiscal_year=None, section=None):
                results = self.retrieve(query, top_k, ticker, fiscal_year, section)
                context_parts = []
                sources = []
                for i, r in enumerate(results, 1):
                    meta = r.get("metadata", {})
                    label = f"[{meta.get('ticker', '?')}/{meta.get('fiscal_year', '?')}/{meta.get('section', '?')}]"
                    context_parts.append(f"--- Source {i} {label} ---\n{r['text']}")
                    sources.append({
                        "chunk_id": r["chunk_id"],
                        "ticker": meta.get("ticker"),
                        "fiscal_year": meta.get("fiscal_year"),
                        "section": meta.get("section"),
                        "score": r["score"],
                    })
                return {
                    "query": query,
                    "num_results": len(results),
                    "context_chunks": results,
                    "context_text": "\n\n".join(context_parts),
                    "sources": sources,
                }

        return RerankedRetriever(base_retriever, reranker, fetch_k=20)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {VALID_MODES}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEC 10-K RAG System")
    parser.add_argument(
        "--mode",
        type=str,
        default="dense",
        choices=VALID_MODES,
        help="Retrieval mode (default: dense)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help="Number of questions to sample (default 15)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run evaluation on all 150 questions",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay in seconds between LLM API calls (default 1.5)",
    )
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    mode = args.mode

    print("=" * 60)
    print(f"  SEC 10-K RAG EVALUATION — MODE: {mode.upper()}")
    print("=" * 60)

    retriever = build_retriever(mode)

    from src.evaluation.evaluator import RAGEvaluator
    from src.generation.answer import NaiveRAGGenerator

    generator = NaiveRAGGenerator(retriever=retriever)
    evaluator = RAGEvaluator(
        retriever=retriever,
        generator=generator,
        scorecard_name=mode,
    )

    summary = evaluator.run_evaluation(
        sample_size=sample_size,
        delay_between_queries=args.delay,
    )

    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY — {mode.upper()}")
    print("=" * 60)
    print(f"Questions Evaluated: {summary['total_questions_evaluated']}")
    print(f"Execution Time:      {summary['execution_time_seconds']}s")
    print(f"Model Used:          {summary['model_used']}")
    print("-" * 60)
    m = summary["metrics"]
    print(f"Retrieval Hit@5:     {m['retrieval_hit_at_5'] * 100:.1f}%")
    print(f"Retrieval MRR@5:     {m['retrieval_mrr_at_5'] * 100:.1f}%")
    print(f"Citation Rate:       {m['citation_rate'] * 100:.1f}%")
    print(f"Refusal Accuracy:    {m['refusal_accuracy'] * 100:.1f}%")
    print(f"Numeric Overlap:     {m['avg_numeric_overlap'] * 100:.1f}%")
    print("=" * 60)
    print(f"\nScorecard: data/eval/scorecard_{mode}.md")


if __name__ == "__main__":
    main()
