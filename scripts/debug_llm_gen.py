import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever
from src.generation.answer import NaiveRAGGenerator
from src.agent.grader import DocumentGrader, HallucinationGrader

retriever = HybridRetriever()
chunks = retriever.retrieve("What was Apple's net income for fiscal year 2024?", ticker="AAPL", fiscal_year=2024, top_k=5)

print(f"Retrieved {len(chunks)} chunks:")
for i, c in enumerate(chunks, 1):
    m = c.get("metadata", {})
    print(f"[{i}] [{m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}]")

doc_grader = DocumentGrader()
relevant = doc_grader.filter_relevant_chunks("What was Apple's net income for fiscal year 2024?", chunks)
print(f"\nFiltered Relevant Chunks count: {len(relevant)}")

generator = NaiveRAGGenerator()
res = generator.answer_from_chunks("What was Apple's net income for fiscal year 2024?", chunks=chunks)
print("\n--- Direct Generator Output ---")
print(res.get("answer"))

if relevant:
    res_rel = generator.answer_from_chunks("What was Apple's net income for fiscal year 2024?", chunks=relevant)
    print("\n--- Generator Output from Relevant Chunks Only ---")
    print(res_rel.get("answer"))
