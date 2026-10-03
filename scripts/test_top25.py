import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.answer import NaiveRAGGenerator

retriever = HybridRetriever()
reranker = CrossEncoderReranker()

q = "What was Apple's net income for fiscal year 2024?"

# Test top_k=25 candidate pool
candidates = retriever.retrieve(q, ticker="AAPL", fiscal_year=2024, top_k=25)
print(f"Retrieved {len(candidates)} candidates.")

# Rerank to top 10
reranked = reranker.rerank(q, candidates, top_k=10)
print(f"\nReranked Top 10 Chunks:")
for i, c in enumerate(reranked, 1):
    m = c.get("metadata", {})
    sec = m.get("section", "N/A")
    score = c.get("rerank_score", 0.0)
    print(f"[{i}] {sec} (Score: {score:.4f})")
    if "Net income" in c.get("text", ""):
        print(f"   -> CONTAINS 'Net income'!")

generator = NaiveRAGGenerator()
res = generator.answer_from_chunks(q, chunks=reranked[:5])
print("\n--- Generator Answer from Reranked Top 5 ---")
print(res.get("answer"))
