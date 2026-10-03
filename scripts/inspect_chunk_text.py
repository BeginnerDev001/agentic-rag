import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
chunks = retriever.retrieve("What was Apple's net income for fiscal year 2024?", ticker="AAPL", fiscal_year=2024, top_k=10)

print(f"Retrieved {len(chunks)} chunks:")
for i, c in enumerate(chunks, 1):
    txt = c.get("text")
    meta = c.get("metadata", {})
    sec = meta.get("section", "N/A")
    print(f"[{i}] Section: {sec}")
    print(f"    root text type: {type(txt)}, len: {len(txt) if txt is not None else 'None'}")
    print(f"    meta keys: {list(meta.keys())}")
