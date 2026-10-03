import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

hybrid = HybridRetriever()
reranker = Reranker()

q = "What was Apple's net income for fiscal year 2024?"
chunks = hybrid.retrieve(q, candidate_pool_size=20, top_k=20)
reranked = reranker.rerank(q, chunks, top_k=5)

for idx, c in enumerate(reranked, 1):
    m = c.get("metadata", {})
    print(f"--- Chunk #{idx} [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}] (Score: {c.get('rerank_score'):.4f}) ---")
    print(c.get("text"))
    print("\n")
