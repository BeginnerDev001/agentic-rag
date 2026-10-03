import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()

# Test retrieving specifically for section_id or keyword boost
q = "Apple Inc. Consolidated Statements of Operations Net income fiscal 2024"
chunks = retriever.retrieve(q, ticker="AAPL", fiscal_year=2024, top_k=10)

print(f"Retrieved {len(chunks)} chunks with table-targeted query:")
for i, c in enumerate(chunks, 1):
    m = c.get("metadata", {})
    print(f"[{i}] {m.get('section')} | text len: {len(c.get('text', ''))}")
    if "$93,736" in c.get("text", "") or "93,736" in c.get("text", ""):
        print("    -> FOUND EXPLICIT NET INCOME 93,736!")
