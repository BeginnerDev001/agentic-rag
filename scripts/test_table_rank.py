import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import DenseRetriever

bm25 = BM25Retriever()
dense = DenseRetriever()

q = "What was Apple's net income for fiscal year 2024?"

bm25_res = bm25.retrieve(q, ticker="AAPL", fiscal_year=2024, top_k=50)
print(f"BM25 found {len(bm25_res)} chunks for AAPL 2024.")
for idx, c in enumerate(bm25_res):
    if "operations" in str(c.get("metadata", {}).get("section", "")).lower() or "$93,736" in c.get("text", ""):
        print(f"  BM25 Rank #{idx+1}: {c.get('metadata')} (Score: {c.get('score')})")

dense_res = dense.retrieve(q, ticker="AAPL", fiscal_year=2024, top_k=50)
print(f"\nDense found {len(dense_res)} chunks for AAPL 2024.")
for idx, c in enumerate(dense_res):
    if "operations" in str(c.get("metadata", {}).get("section", "")).lower() or "$93,736" in c.get("text", ""):
        print(f"  Dense Rank #{idx+1}: {c.get('metadata')} (Distance: {c.get('similarity_score')})")
