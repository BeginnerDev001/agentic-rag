# Project Status: Self-Correcting Agentic RAG for SEC Filings

> **Current Progress:** ~50% Complete  
> **Phases Completed:** Phase 0 (Data & Benchmark), Phase 1 (Baseline RAG & MCP Server), Phase 2 (Agentic Self-Correction Loop & Evals)  
> **Upcoming Phases:** Phase 3 (Hybrid Search & Reranking), Phase 4 (Guardrails), Phase 5 (LLMOps & Serving), Phase 6/7 (Full Benchmark & Write-up)

---

## 1. Core Structure Status

| Core Structure Component | Status | Details & Verification |
| :--- | :---: | :--- |
| **SEC Dataset** | **Done (100%)** | 56 annual 10-K filings collected for 10 companies (`AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`, `TSLA`, `JPM`, `V`, `WMT`). Parsed into 6,109 structured sections and chunked into **11,127 retrieval chunks** with full metadata provenance. |
| **ChromaDB Vector DB** | **Done (100%)** | Persistent ChromaDB collection storing all 11,127 vector chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dim normalized embeddings, cosine similarity metric). |
| **MCP Server** | **Done (100%)** | FastMCP server implemented in `src/mcp_server.py` exposing 3 standard MCP tools (`search_sec_filings`, `answer_financial_question`, `get_corpus_stats`) via stdio for Claude Desktop and agent environments. |
| **Eval Benchmark Suite** | **Done (100%)** | Hand-curated dataset of 150 questions in `data/eval/questions.jsonl` categorized across Factual, Multi-hop/Comparative, Not-in-corpus (Abstention), and Adversarial queries. |

---

## 2. What Has Been Built Till Now

### Phase 0: Corpus Ingestion, Structuring & Benchmark
- **Configuration & Targets**: `config/companies.yaml` defines the 10 target companies across diverse industries.
- **SEC EDGAR Downloader**: `scripts/download_filings.py` fetches 10-K filings with SEC-compliant User-Agent and rate-limiting, storing HTML and logging audit entries in `data/manifests/filings.jsonl`.
- **Text & iXBRL Cleaner**: `src/preprocessing/cleaner.py` strips inline XBRL tags without losing embedded numbers, normalizes whitespace and typographic characters, and converts raw HTML `<table>` elements into Markdown tables.
- **Hierarchical Section Parser**: `src/ingestion/parser.py` and `scripts/process_filings.py` extract standard 10-K items and break down Item 8 financial statements (Operations, Balance Sheets, Cash Flows, Notes) into 6,109 section files.
- **Structure-Aware Chunker**: `src/preprocessing/chunker.py` and `scripts/chunk_filings.py` keep tables atomic and split narrative sections along sentence boundaries with a 64-token overlap, yielding 11,127 chunks.
- **Benchmark Dataset**: `data/eval/questions.jsonl` contains 150 ground-truth Q&A pairs with source section tracking.

### Phase 1: Baseline RAG & Model Context Protocol (MCP) Server
- **Embedding Model Wrapper**: `src/retrieval/embeddings.py` provides lazy loading of `all-MiniLM-L6-v2`, batch processing, and L2 normalization for fast dot-product similarity.
- **ChromaDB Vector Store**: `src/retrieval/vector_store.py` manages persistent storage, metadata sanitization, and filtered HNSW cosine vector search.
- **Ingestion Pipeline**: `scripts/ingest_chunks.py` and `scripts/ingest_by_ticker.py` handle batch vector indexing across tickers using isolated subprocesses to avoid memory fragmentation.
- **Dense Retriever**: `src/retrieval/retriever.py` provides an interface for vector similarity search with ticker, fiscal year, and section metadata filters.
- **Naive Generator**: `src/generation/answer.py` formats prompt templates with retrieved filing excerpts and generates citations (`[TICKER/YEAR/SECTION]`) using OpenRouter (`openai/gpt-4o-mini`).
- **MCP Server**: `src/mcp_server.py` implements a FastMCP stdio server exposing:
  - `search_sec_filings(query, ticker, fiscal_year, section, top_k)`
  - `answer_financial_question(query, ticker, fiscal_year, section)`
  - `get_corpus_stats()`
- **Baseline v0 Scorecard**: `data/eval/scorecard_v0.md` logs naive RAG baseline metrics (100% Hit@5, 100% Refusal Accuracy, 60% Citation Rate).

### Phase 2: Agentic Layer & Self-Correcting Loop
- **Query Rewriter & Decomposer**: `src/agent/rewriter.py` decomposes multi-company or complex comparative questions into targeted sub-queries and reformulates user language into SEC financial terminology.
- **Self-Critique Graders**: `src/agent/grader.py` implements:
  - `DocumentGrader`: Assesses document relevance to strip out irrelevant boilerplates.
  - `HallucinationGrader`: Verifies whether generated answers are strictly grounded in retrieved evidence.
  - `AnswerQualityGrader`: Validates prompt utility.
- **Agentic Orchestrator**: `src/agent/agent.py` implements a self-correction state loop:
  - Decomposes multi-part queries.
  - Formulates optimized search queries.
  - Grades retrieved chunks; if no relevant chunks are found, rewrites the query and retries (up to 2-3 attempts).
  - Generates answers and verifies factuality; if hallucinations are detected, re-prompts with strict factuality constraints.
  - Safely abstains with a calibrated refusal when evidence is insufficient.
- **Comparative Scorecard v1**: `data/eval/scorecard_v1.md` tracks improvements from Baseline v0 to Agentic v1.

---

## 3. What the Project Is Capable of Doing Right Now

1. **Semantic Similarity Search Over SEC Filings**:
   - Querying 11,127 pre-chunked 10-K sections by topic or question.
   - Exact filtering by ticker (e.g. `AAPL`), fiscal year (e.g. `2024`), or section type.
2. **Naive Single-Shot RAG Generation**:
   - Retrieving top-$k$ chunks, packing them into an LLM context, and producing an answer with source citations.
3. **MCP Tool Serving**:
   - Connecting directly to Claude Desktop, IDE extensions, or agent clients via stdio MCP to let models search filings and inspect corpus stats.
4. **Autonomous Self-Correcting Agentic Inquiries**:
   - Decomposing comparative questions (e.g., comparing Apple vs Microsoft cloud growth).
   - Filtering noise and boilerplate disclosures.
   - Retrying with reformulated queries if retrieval is insufficient.
   - Refusing to answer when facts are absent (e.g., out-of-corpus queries).
5. **Standardized Evaluation**:
   - Running test suites and scoring pipelines on Hit@k, MRR@k, Citation Rate, Refusal Accuracy, and Numeric Overlap.

---

## 4. What Is Left to Be Made

The project has completed its baseline and self-correcting agent phases. The remaining work covers precision enhancement, safety, deployment, and benchmark documentation:

### Phase 3: Hybrid Search & Reranking (Next Up)
- **BM25 Lexical Index**: SEC filings contain exact financial terms (e.g., *"Form 10-K/A Item 8"*, *"ASC 606"*, exact dollar figures) where semantic embeddings sometimes struggle. Add BM25 sparse keyword retrieval.
- **Reciprocal Rank Fusion (RRF)**: Combine dense semantic rankings with sparse BM25 rankings.
- **Cross-Encoder Reranker**: Integrate a high-precision reranker model (e.g. `bge-reranker-large` or fine-tuned LoRA model) to score the top-25 merged candidates down to the top-5 most relevant chunks.

### Phase 4: Safety & Guardrails
- **Input Guardrails**: Prompt-injection / jailbreak detection classifier to prevent system overrides and prompt leakage (evaluating against the 10 adversarial benchmark questions).
- **Scope Enforcement**: Fast intent classifier rejecting non-financial / off-topic queries before touching vector or LLM resources.
- **Output Guardrails & Citation Verifier**: Strict citation checker verifying that every claimed number in the answer exists character-for-character in the retrieved context; append financial advice disclaimers.

### Phase 5: LLMOps Polish & Production Serving
- **Caching Layer**: In-memory or Redis caching for identical and semantically near queries to reduce API latency and cost.
- **Tracing & Observability**: Integration with Langfuse or Weights & Biases to track query latency, token usage, retry count, and costs per stage.
- **FastAPI Web Service**: REST API wrapping the agent pipeline alongside the MCP server.
- **Dockerization**: `Dockerfile` and `docker-compose.yml` for single-command reproducible container deployment.
- **CI/CD Eval Hook**: Automated lightweight eval run on git push to prevent regression.

### Phase 6 / 7: Comprehensive Benchmark & Portfolio Write-Up
- **Full 150-Question Benchmark**: Run complete evaluation comparing:
  - Baseline v0 (Naive RAG)
  - Agentic v1 (Self-Correction Loop)
  - Hybrid/Reranked v2 (Dense + BM25 + Cross-Encoder)
  - Guardrailed v3
- **Final Portfolio Documentation**: Architecture diagrams, cost/latency trade-off analysis, and eval comparison table front and center in `README.md`.
