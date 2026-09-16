"""
Test script for verifying DocumentGrader, HallucinationGrader, and AnswerQualityGrader.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.grader import AnswerQualityGrader, DocumentGrader, HallucinationGrader


def main():
    print("=" * 60)
    print("  SELF-CRITIQUE GRADERS - VERIFICATION TEST")
    print("=" * 60)

    doc_grader = DocumentGrader()
    hallucination_grader = HallucinationGrader()
    answer_grader = AnswerQualityGrader()

    # Test 1: DocumentGrader
    print("\n--- Test 1: DocumentGrader ---")
    query1 = "What was Apple's total net sales in 2024?"
    rel_doc = "Apple Inc. Consolidated Statements of Operations. Net sales in 2024 were $391,035 million."
    irrel_doc = "Item 1. Business. Apple designs a wide variety of consumer electronics including Mac and iPad."

    rel_is, rel_reason = doc_grader.grade(query1, rel_doc)
    irrel_is, irrel_reason = doc_grader.grade(query1, irrel_doc)

    print(f"Relevant Doc Grade:   {rel_is} | Reason: {rel_reason}")
    print(f"Irrelevant Doc Grade: {irrel_is} | Reason: {irrel_reason}")

    # Test 2: HallucinationGrader
    print("\n--- Test 2: HallucinationGrader ---")
    ctx = "Apple Inc. reported net income of $93.7 billion in fiscal year 2024."
    grounded_ans = "Apple's net income for fiscal year 2024 was $93.7 billion [AAPL/2024/Item 8]."
    hallucinated_ans = "Apple's net income was $150 billion, making it the most profitable company in history."

    g_is, g_reason = hallucination_grader.grade(ctx, grounded_ans)
    h_is, h_reason = hallucination_grader.grade(ctx, hallucinated_ans)

    print(f"Grounded Answer Grade:     {g_is} | Reason: {g_reason}")
    print(f"Hallucinated Answer Grade: {h_is} | Reason: {h_reason}")

    # Test 3: AnswerQualityGrader
    print("\n--- Test 3: AnswerQualityGrader ---")
    query3 = "What was Microsoft's R&D expenditure in 2024?"
    useful_ans = "Microsoft spent $29.5 billion on research and development in fiscal year 2024 [MSFT/2024/Item 8]."

    u_is, u_reason = answer_grader.grade(query3, useful_ans)
    print(f"Answer Quality Grade: {u_is} | Reason: {u_reason}")

    print("\n" + "=" * 60)
    print("  SELF-CRITIQUE GRADERS TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
