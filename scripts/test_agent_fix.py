import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.agent import AgenticRAG

agent = AgenticRAG()

print("=== TEST 1: Apple FY2024 Net Income ===")
res1 = agent.ask("What was Apple's net income for fiscal year 2024?")
print("Answer:", res1.get("answer"))
print("Status:", res1.get("status"))
print("Sources count:", len(res1.get("sources", [])))

print("\n=== TEST 2: Microsoft FY2023 R&D Spend ===")
res2 = agent.ask("How much did Microsoft spend on research and development in fiscal year 2023?")
print("Answer:", res2.get("answer"))
print("Status:", res2.get("status"))
print("Sources count:", len(res2.get("sources", [])))
