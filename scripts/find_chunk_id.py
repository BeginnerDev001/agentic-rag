import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import VectorStore

bm25 = BM25Retriever()
vs = VectorStore()

target_id = "AAPL_2024_item_8_operations_000_445e1a71"

# Check BM25
found_bm25 = None
for c in bm25.chunks:
    if c.get("chunk_id") == target_id or target_id in c.get("chunk_id", ""):
        found_bm25 = c
        break

print("=== BM25 FOUND ===", found_bm25 is not None)
if found_bm25:
    print("BM25 metadata:", found_bm25.get("metadata"))
    print("BM25 text snippet:\n", found_bm25.get("text", "")[:300])

# Check ChromaDB
res = vs.collection.get(ids=[target_id])
print("\n=== CHROMADB FOUND ===", bool(res and res.get("ids")))
if res and res.get("ids"):
    print("Chroma metadata:", res.get("metadatas"))
    print("Chroma text snippet:\n", res.get("documents")[0][:300] if res.get("documents") else "NO DOC")
