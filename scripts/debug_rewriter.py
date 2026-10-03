import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.rewriter import QueryRewriter

rewriter = QueryRewriter()

q_aapl = "What was Apple's net income for fiscal year 2024?"
rw_aapl = rewriter.rewrite(q_aapl)
print("AAPL original:", q_aapl)
print("AAPL rewritten:", rw_aapl)

q_msft = "How much did Microsoft spend on research and development in fiscal year 2023?"
rw_msft = rewriter.rewrite(q_msft)
print("\nMSFT original:", q_msft)
print("MSFT rewritten:", rw_msft)
