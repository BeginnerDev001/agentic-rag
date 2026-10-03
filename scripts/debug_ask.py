import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.agent import AgenticRAG

agent = AgenticRAG()
res = agent.ask("What was Apple's net income for fiscal year 2024?")
print("--- RESULT ---")
print("Answer:", res.get("answer"))
print("Status:", res.get("status"))
print("Sources count:", len(res.get("sources", [])))
print("Sources sample:", res.get("sources", [])[:2])
print("Trace:", res.get("execution_trace"))
