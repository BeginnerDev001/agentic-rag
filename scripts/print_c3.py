import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
achunks = retriever.retrieve("What was Apple's net income for fiscal year 2024?", ticker="AAPL", fiscal_year=2024, top_k=10)

c3 = achunks[2]
print("=== CHUNK 3 SECTION ===", c3.get("metadata", {}).get("section"))
print("=== CHUNK 3 TEXT ===")
print(c3.get("text"))
