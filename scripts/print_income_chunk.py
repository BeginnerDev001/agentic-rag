import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
candidates = retriever.retrieve("What was Apple's net income for fiscal year 2024?", ticker="AAPL", fiscal_year=2024, top_k=25)

for c in candidates:
    m = c.get("metadata", {})
    sec = m.get("section", "")
    if "comprehensive" in sec.lower() or "operations" in sec.lower():
        print("=== FOUND SECTION:", sec, "===")
        print(c.get("text"))
        print("\n" + "="*50 + "\n")
