import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG

agent = AgenticRAG()
res = agent.ask("What was Apple's net income for fiscal year 2024?")

print("=== APPLE SOURCE CHUNK ===")
for s in res.get("sources", []):
    print("Section:", s.get("section"))
    print("Full Text:\n", repr(s.get("text")))
