"""
CLI script to run baseline evaluation on SEC 10-K RAG dataset.

Usage:
  python scripts/evaluate.py --sample 20
  python scripts/evaluate.py --full
"""

import argparse
import os
import sys
from pathlib import Path

# Force CPU single-thread for PyTorch/OpenMP safety on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEC 10-K RAG System")
    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help="Number of questions to sample for evaluation (default 15)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run evaluation on all 150 questions in dataset",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay in seconds between LLM queries to respect API rate limits",
    )
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    print("=" * 60)
    print("  SEC 10-K RAG SYSTEM - BASELINE V0 EVALUATION")
    print("=" * 60)
    if args.full:
        print("Running full dataset (150 questions)...")
    else:
        print(f"Running sample of {sample_size} questions...")

    evaluator = RAGEvaluator()
    summary = evaluator.run_evaluation(
        sample_size=sample_size,
        delay_between_queries=args.delay,
    )

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY SCORECARD")
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


if __name__ == "__main__":
    main()
