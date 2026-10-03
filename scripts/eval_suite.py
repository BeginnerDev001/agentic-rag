"""
Automated RAG Evaluation Suite for Agentic Financial RAG System.

Runs a curated benchmark of SEC 10-K financial queries, measures:
  - Retrieval Precision (relevant chunks / total retrieved)
  - Answer Accuracy (expected keywords/numbers found in answer)
  - Abstention Accuracy (correct rejection of out-of-scope queries)
  - Latency (seconds per query)

Outputs: scorecard_v3.md  +  eval_results.json
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.agent import AgenticRAG

# ──────────────────────────────────────────────────────────
# Benchmark Test Cases
# ──────────────────────────────────────────────────────────
TEST_CASES = [
    # ── Quantitative Financial Metrics ──
    {
        "id": "Q1",
        "query": "What was Apple's net income for fiscal year 2024?",
        "expected_status": "success",
        "expected_keywords": ["93,736", "93736", "net income"],
        "expected_ticker": "AAPL",
        "category": "quantitative",
    },
    {
        "id": "Q2",
        "query": "How much did Microsoft spend on research and development in fiscal year 2023?",
        "expected_status": "success",
        "expected_keywords": ["27,195", "27195", "research and development"],
        "expected_ticker": "MSFT",
        "category": "quantitative",
    },
    {
        "id": "Q3",
        "query": "What was Apple's total net sales for fiscal year 2024?",
        "expected_status": "success",
        "expected_keywords": ["391,035", "391035", "net sales"],
        "expected_ticker": "AAPL",
        "category": "quantitative",
    },
    {
        "id": "Q4",
        "query": "What was Microsoft's total revenue for fiscal year 2023?",
        "expected_status": "success",
        "expected_keywords": ["211,915", "211915", "revenue"],
        "expected_ticker": "MSFT",
        "category": "quantitative",
    },
    # ── Qualitative / MD&A Queries ──
    {
        "id": "Q5",
        "query": "What are the key risk factors mentioned in Apple's 2024 10-K filing?",
        "expected_status": "success",
        "expected_keywords": ["risk"],
        "expected_ticker": "AAPL",
        "category": "qualitative",
    },
    {
        "id": "Q6",
        "query": "Describe Microsoft's business segments as reported in their 2023 10-K.",
        "expected_status": "success",
        "expected_keywords": ["segment", "cloud", "productivity"],
        "expected_ticker": "MSFT",
        "category": "qualitative",
    },
    # ── Out-of-Scope / Abstention Queries ──
    {
        "id": "Q7",
        "query": "What is the current price of Bitcoin?",
        "expected_status": "abstained",
        "expected_keywords": ["insufficient", "not", "cannot", "outside"],
        "expected_ticker": None,
        "category": "out_of_scope",
    },
    {
        "id": "Q8",
        "query": "Who won the 2024 FIFA World Cup?",
        "expected_status": "abstained",
        "expected_keywords": ["insufficient", "not", "cannot", "outside"],
        "expected_ticker": None,
        "category": "out_of_scope",
    },
]


def check_answer_accuracy(answer_text: str, expected_keywords: list) -> bool:
    """Check if any expected keyword appears in the answer (case-insensitive)."""
    answer_lower = answer_text.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def check_abstention(answer_text: str, status: str) -> bool:
    """Check if the system correctly abstained."""
    abstention_signals = ["insufficient", "cannot", "not available", "outside", "guardrail"]
    answer_lower = answer_text.lower()
    is_abstained = any(s in answer_lower for s in abstention_signals)
    is_guardrail = status in ("guardrail_rejected", "abstained")
    return is_abstained or is_guardrail


def run_evaluation():
    """Run all benchmark test cases and compute metrics."""
    print("=" * 70)
    print("  AGENTIC FINANCIAL RAG - AUTOMATED EVALUATION SUITE")
    print("=" * 70)

    agent = AgenticRAG()

    results = []
    total_correct = 0
    total_retrieval_precision = 0.0
    total_latency = 0.0
    abstention_correct = 0
    abstention_total = 0

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{tc['id']}] {tc['query']}")
        print("-" * 60)

        start = time.time()
        try:
            res = agent.ask(tc["query"])
        except Exception as e:
            res = {"answer": f"ERROR: {e}", "status": "error", "sources": []}
        elapsed = time.time() - start

        answer = res.get("answer", "")
        status = res.get("status", "unknown")
        sources = res.get("sources", [])
        num_chunks = len(sources)

        # ── Metrics ──
        if tc["expected_status"] == "abstained":
            abstention_total += 1
            is_correct = check_abstention(answer, status)
            if is_correct:
                abstention_correct += 1
                total_correct += 1
            precision = 0.0  # N/A for abstention
        else:
            is_correct = check_answer_accuracy(answer, tc["expected_keywords"])
            if is_correct:
                total_correct += 1
            # Retrieval precision: count sources matching expected ticker
            if tc["expected_ticker"] and num_chunks > 0:
                relevant = sum(1 for s in sources if s.get("ticker") == tc["expected_ticker"])
                precision = relevant / num_chunks
            else:
                precision = 1.0 if num_chunks > 0 else 0.0

        total_retrieval_precision += precision
        total_latency += elapsed

        verdict = "PASS" if is_correct else "FAIL"
        print(f"  Answer: {answer[:120]}...")
        print(f"  Status: {status} | Chunks: {num_chunks} | Precision: {precision:.2f} | Latency: {elapsed:.1f}s")
        print(f"  Verdict: {'✅' if is_correct else '❌'} {verdict}")

        results.append({
            "id": tc["id"],
            "query": tc["query"],
            "category": tc["category"],
            "expected_status": tc["expected_status"],
            "actual_status": status,
            "answer_snippet": answer[:200],
            "is_correct": is_correct,
            "retrieval_precision": round(precision, 3),
            "num_chunks": num_chunks,
            "latency_seconds": round(elapsed, 2),
        })

    # ── Summary ──
    n = len(TEST_CASES)
    non_abstention = n - abstention_total
    answer_accuracy = total_correct / n * 100 if n else 0
    avg_precision = total_retrieval_precision / n * 100 if n else 0
    avg_latency = total_latency / n if n else 0
    abstention_acc = abstention_correct / abstention_total * 100 if abstention_total else 100

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total Test Cases:        {n}")
    print(f"  Overall Accuracy:        {total_correct}/{n} ({answer_accuracy:.1f}%)")
    print(f"  Avg Retrieval Precision: {avg_precision:.1f}%")
    print(f"  Abstention Accuracy:     {abstention_correct}/{abstention_total} ({abstention_acc:.1f}%)")
    print(f"  Avg Latency:             {avg_latency:.1f}s")
    print("=" * 70)

    # ── Save JSON ──
    eval_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_cases": n,
            "overall_accuracy_pct": round(answer_accuracy, 1),
            "avg_retrieval_precision_pct": round(avg_precision, 1),
            "abstention_accuracy_pct": round(abstention_acc, 1),
            "avg_latency_seconds": round(avg_latency, 1),
        },
        "results": results,
    }
    json_path = PROJECT_ROOT / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # ── Generate Scorecard ──
    scorecard_path = PROJECT_ROOT / "scorecard_v3.md"
    generate_scorecard(scorecard_path, eval_output, results)
    print(f"  Saved: {scorecard_path}")

    return eval_output


def generate_scorecard(path: Path, summary: dict, results: list):
    """Generate scorecard_v3.md markdown file."""
    s = summary["summary"]
    lines = [
        "# Scorecard v3 — Automated RAG Evaluation",
        "",
        f"**Date**: {summary['timestamp']}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Test Cases | {s['total_cases']} |",
        f"| Overall Accuracy | {s['overall_accuracy_pct']}% |",
        f"| Avg Retrieval Precision | {s['avg_retrieval_precision_pct']}% |",
        f"| Abstention Accuracy | {s['abstention_accuracy_pct']}% |",
        f"| Avg Latency | {s['avg_latency_seconds']}s |",
        "",
        "## Detailed Results",
        "",
        "| ID | Category | Query | Correct | Precision | Latency |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        q_short = r["query"][:50] + ("..." if len(r["query"]) > 50 else "")
        verdict = "✅" if r["is_correct"] else "❌"
        lines.append(
            f"| {r['id']} | {r['category']} | {q_short} | {verdict} | {r['retrieval_precision']:.0%} | {r['latency_seconds']}s |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/eval_suite.py`*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
