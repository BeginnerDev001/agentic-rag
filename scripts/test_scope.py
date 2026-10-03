import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.abstention import FinancialScopeValidator
from src.retrieval.hybrid import HybridRetriever

validator = FinancialScopeValidator()
q = "What was Apple's net income for fiscal year 2024?"
scope = validator.extract_scope(q)
print("Extracted Scope:", scope)

retriever = HybridRetriever()
chunks = retriever.retrieve(q, ticker=scope["ticker"], fiscal_year=scope["fiscal_year"], top_k=10)

print(f"\nRetrieved {len(chunks)} chunks for {scope}:")
for i, c in enumerate(chunks, 1):
    m = c.get("metadata", {})
    print(f"[{i}] {m.get('ticker')}/{m.get('fiscal_year')}/{m.get('section')}: {c.get('text')[:120]}...")
