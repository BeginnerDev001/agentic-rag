import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()

print("=== MSFT RETRIEVAL DEBUG ===")
chunks = retriever.retrieve("How much did Microsoft spend on research and development in fiscal year 2023?", ticker="MSFT", fiscal_year=2023, top_k=10)
for i, c in enumerate(chunks, 1):
    m = c.get("metadata", {})
    sec = m.get("section", "N/A")
    ctype = m.get("content_type", "N/A")
    rrf = c.get("rrf_score")
    txt = c.get("text", "")[:100].replace("\n", " ")
    print(f"[{i}] {sec} | ctype: {ctype} | RRF: {rrf} | snippet: {txt}")

print("\n=== AAPL RETRIEVAL DEBUG ===")
achunks = retriever.retrieve("What was Apple's net income for fiscal year 2024?", ticker="AAPL", fiscal_year=2024, top_k=10)
for i, c in enumerate(achunks, 1):
    m = c.get("metadata", {})
    sec = m.get("section", "N/A")
    ctype = m.get("content_type", "N/A")
    rrf = c.get("rrf_score")
    txt = c.get("text", "")[:100].replace("\n", " ")
    print(f"[{i}] {sec} | ctype: {ctype} | RRF: {rrf} | snippet: {txt}")
