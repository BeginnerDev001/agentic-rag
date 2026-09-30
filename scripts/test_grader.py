import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.grader import DocumentGrader
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

hybrid = HybridRetriever()
reranker = Reranker()
grader = DocumentGrader()

q = "What was Apple's total net sales in fiscal year 2024?"
print("Step 1: Retrieve top 20 candidate chunks from Hybrid retriever")
candidates = hybrid.retrieve(q, candidate_pool_size=20, top_k=20)

print("\nStep 2: Rerank top 20 candidates down to top 5 using CrossEncoder")
reranked_chunks = reranker.rerank(q, candidates, top_k=5)

for c in reranked_chunks:
    m = c.get("metadata", {})
    score = c.get("rerank_score", 0.0)
    print(f"- [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] (Rerank score: {score:.4f}) {c.get('text')[:100]}...")

print("\nStep 3: Grade relevant chunks with DocumentGrader")
rel_chunks = grader.filter_relevant_chunks(q, reranked_chunks)
print(f"Relevant chunks after grading: {len(rel_chunks)}")
for r in rel_chunks:
    print(f"  Reason: {r.get('relevance_reason')}")
