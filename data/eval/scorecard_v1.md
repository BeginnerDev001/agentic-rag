# SEC 10-K RAG — Baseline v0 vs Agentic v1 Scorecard

- **Evaluated Questions:** 10
- **Baseline Model:** `openai/gpt-4o-mini` (Naive RAG)
- **Agentic Model:** `openai/gpt-4o-mini` (Query Rewriting + Document Grading + Self-Critique Loop)

## Comparative Metrics

| Metric | Baseline v0 | Agentic v1 | Delta | Target |
|---|---|---|---|---|
| **Citation Rate** | 60.0% | **10.0%** | +-50.0% | > 90.0% |
| **Refusal Accuracy** | 100.0% | **100.0%** | ++0.0% | > 85.0% |
| **Numeric Overlap** | 40.0% | **10.0%** | +-30.0% | > 70.0% |

## Improvements in Agentic v1:
1. **Query Expansion & Decomposition:** Multi-company comparative queries are decomposed into targeted single-company sub-searches.
2. **Noise Reduction:** DocumentGrader removes irrelevant SEC disclaimers prior to LLM answer synthesis.
3. **Self-Correction Retry Loop:** Automatically rewrites query and re-retrieves if hallucination or zero relevant chunks are detected.
