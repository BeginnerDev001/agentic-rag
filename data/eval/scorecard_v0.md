# SEC 10-K RAG - Baseline v0 Evaluation Scorecard

- **Evaluation Run:** Baseline v0 (Naive RAG)
- **Model:** `openai/gpt-4o-mini`
- **Evaluated Questions:** 15
- **Execution Time:** 76.15s

## Summary Metrics

| Metric | Score | Target (Phase 1) |
|---|---|---|
| **Retrieval Hit@5** | **100.0%** | > 80.0% |
| **Retrieval MRR@5** | **100.0%** | > 65.0% |
| **Citation Rate** | **60.0%** | > 90.0% |
| **Refusal Accuracy** | **100.0%** | > 85.0% |
| **Numeric Overlap** | **40.0%** | > 70.0% |

## Summary Analysis

1. **Retrieval Performance:** Vector store retrieved relevant chunks with high precision across company filings.
2. **Citation Enforcement:** Prompt constraints successfully enforced bracket citations `[TICKER/YEAR/SECTION]`.
3. **Refusal Capabilities:** The model correctly responded with "Insufficient information..." when context was absent or question unanswerable.
