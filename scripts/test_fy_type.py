import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever

bm25 = BM25Retriever()
res_int = bm25.retrieve("research and development", ticker="MSFT", fiscal_year=2023)
res_str = bm25.retrieve("research and development", ticker="MSFT", fiscal_year="2023")

print("BM25 results with fiscal_year=2023 (int):", len(res_int))
print("BM25 results with fiscal_year='2023' (str):", len(res_str))
if res_int:
    print("Sample int:", res_int[0]["metadata"])
