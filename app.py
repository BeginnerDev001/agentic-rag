"""
Streamlit Interactive Dashboard for Agentic Financial RAG.

Run locally via:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

# Prevent OpenMP / PyTorch Windows process crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG
from src.guardrails.abstention import FinancialScopeValidator

# ─── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Financial RAG — SEC 10-K Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Agentic Financial RAG — SEC 10-K Intelligence")
st.markdown(
    "Hybrid Lexical + Vector Retrieval with Cross-Encoder Reranking & Self-Correcting LLM Guardrails over 11,000+ SEC Filing Chunks."
)

# ─── Cache Agent Initialization ───────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Agentic RAG System (Loading BM25 & Embeddings)...")
def get_agent():
    return AgenticRAG(use_reranker=True)

agent = get_agent()
scope_validator = FinancialScopeValidator()

# ─── Sidebar Controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Query Configuration")
    
    ticker_option = st.selectbox(
        "Filter by Company Ticker",
        options=["All", "AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "V", "WMT"],
        index=0,
    )
    
    year_option = st.selectbox(
        "Filter by Fiscal Year",
        options=["All"] + list(range(2025, 2015, -1)),
        index=0,
    )

    max_retries = st.slider("Max Self-Correction Retries", min_value=1, max_value=4, value=2)

    st.markdown("---")
    st.subheader("📊 Corpus Details")
    st.caption("• **11,127 Text Chunks**")
    st.caption("• **10 S&P 500 Tech & Financial Giants**")
    st.caption("• **Years Covered**: 2016 – 2025")
    st.caption("• **Retrieval**: Hybrid (Dense ChromaDB + BM25 RRF)")
    st.caption("• **Reranker**: Cross-Encoder MS-MARCO")

# ─── Sample Queries ──────────────────────────────────────────────────
st.subheader("💡 Try Example Financial Queries")
col1, col2, col3 = st.columns(3)

sample_q = ""
if col1.button("🍎 Apple FY2024 Net Income"):
    sample_q = "What was Apple's net income for fiscal year 2024?"
if col2.button("💻 Microsoft R&D Spend 2023"):
    sample_q = "How much did Microsoft spend on research and development in fiscal year 2023?"
if col3.button("☁️ Amazon vs Google Cloud Revenue"):
    sample_q = "What was AWS revenue compared to Google Cloud revenue in fiscal year 2024?"

# ─── Main Input ──────────────────────────────────────────────────────
query_input = st.text_input(
    "Ask a question about SEC 10-K filings:",
    value=sample_q if sample_q else "",
    placeholder="e.g., What was Apple's total net sales in fiscal year 2024?",
)

if st.button("🚀 Run Agentic RAG", type="primary") or sample_q:
    if not query_input.strip():
        st.warning("Please enter a financial question.")
    else:
        # Resolve sidebar metadata filters
        ticker_filter = None if ticker_option == "All" else ticker_option
        year_filter = None if year_option == "All" else int(year_option)

        # Update agent max retries
        agent.max_retries = max_retries

        # Execute query with progress spinner
        start_time = time.time()
        with st.spinner("Agent searching SEC 10-K filings, grading documents, and synthesizing answer..."):
            result = agent.ask(
                query=query_input,
                ticker=ticker_filter,
                fiscal_year=year_filter,
            )
        elapsed = time.time() - start_time

        st.markdown("---")
        
        # ─── Answer Box ──────────────────────────────────────────────
        answer_text = result.get("answer", "No answer returned.")
        status = result.get("status", "unknown")
        
        if status == "guardrail_rejected":
            st.error(f"🛡️ **Guardrail Triggered**: {answer_text}")
        elif "insufficient information" in answer_text.lower():
            st.warning(f"⚠️ **Abstention**: {answer_text}")
        else:
            st.success("### 📜 Verified Answer")
            st.markdown(answer_text)

        st.caption(f"⏱️ **Execution Time**: {elapsed:.2f} seconds | 🔄 **Retries**: {result.get('retries', 0)}")

        # ─── Tabs for Trace & Sources ────────────────────────────────
        tab_sources, tab_trace, tab_guardrails = st.tabs([
            "📚 Cited Sources & Chunks",
            "🔄 Agent Execution Trace",
            "🛡️ Guardrails & Scope",
        ])

        with tab_sources:
            sources = result.get("sources", [])
            if not sources:
                st.info("No sources retrieved or cited.")
            else:
                st.markdown(f"Found **{len(sources)}** relevant passage chunks:")
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    t = src.get("ticker") or meta.get("ticker", "N/A")
                    fy = src.get("fiscal_year") or meta.get("fiscal_year", "N/A")
                    sec = src.get("section") or meta.get("section", "N/A")
                    score = src.get("rerank_score") or src.get("similarity_score") or src.get("score", 0.0)
                    text = src.get("text") or meta.get("text", "Text cited from retrieved SEC filing chunk.")
                    with st.expander(
                        f"Chunk #{idx} — [{t}/{fy}/{sec}] (Score: {score:.4f})"
                    ):
                        st.markdown(f"**Section**: `{sec}`")
                        st.text_area("Passage Text", value=text, height=150, key=f"src_{idx}")

        with tab_trace:
            trace = result.get("execution_trace", [])
            if not trace:
                st.info("No execution trace recorded.")
            else:
                for step in trace:
                    st.json(step)

        with tab_guardrails:
            scope_info = scope_validator.extract_scope(query_input)
            st.write("**Detected Scope from Query:**")
            st.json(scope_info)

st.markdown("---")
st.caption("Agentic Financial RAG System | Built with Streamlit, ChromaDB, BM25, SentenceTransformers, and OpenRouter LLM")
