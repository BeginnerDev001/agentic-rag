import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG

agent = AgenticRAG()
res = agent.ask("How much did Microsoft spend on research and development in fiscal year 2023?")
print("--- MSFT 2023 R&D RESULT ---")
print("Answer:\n", res.get("answer"))
print("\nStatus:", res.get("status"))
print("Sources count:", len(res.get("sources", [])))
for s in res.get("sources", []):
    print("Source:", s)
