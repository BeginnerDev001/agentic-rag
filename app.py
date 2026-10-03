"""
Streamlit Interactive Dashboard for Agentic Financial RAG with Multi-Turn Chat.

Features:
- Multi-Turn Conversational Memory (st.chat_message & st.chat_input)
- Hybrid Lexical + Vector RAG over 11,000+ SEC Filing Chunks
- Self-Correcting LLM Guardrails & Query Rewriting
- Interactive Plotly Financial Charts & Multi-Year Trends
- Key Financial Ratios Calculator (Net Margin %, R&D / Revenue %)

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
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG
from src.guardrails.abstention import FinancialScopeValidator

# ─── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Financial RAG — Multi-Turn SEC Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Agentic Financial RAG — Multi-Turn SEC Intelligence")
st.markdown(
    "Multi-Turn Conversational RAG over 11,000+ SEC Filing Chunks with Hybrid Vector/BM25 Search & Interactive Financial Analytics."
)

# ─── SEC Financial Benchmark Dataset ─────────────────────────────────
FINANCIAL_DATA = [
    {"Ticker": "AAPL", "Company": "Apple Inc.", "Year": 2021, "Revenue_M": 365817, "NetIncome_M": 94680, "RD_M": 21914},
    {"Ticker": "AAPL", "Company": "Apple Inc.", "Year": 2022, "Revenue_M": 394328, "NetIncome_M": 99803, "RD_M": 26251},
    {"Ticker": "AAPL", "Company": "Apple Inc.", "Year": 2023, "Revenue_M": 383285, "NetIncome_M": 96995, "RD_M": 29915},
    {"Ticker": "AAPL", "Company": "Apple Inc.", "Year": 2024, "Revenue_M": 391035, "NetIncome_M": 93736, "RD_M": 31370},
    
    {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Year": 2021, "Revenue_M": 168088, "NetIncome_M": 61271, "RD_M": 20716},
    {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Year": 2022, "Revenue_M": 198270, "NetIncome_M": 72738, "RD_M": 24512},
    {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Year": 2023, "Revenue_M": 211915, "NetIncome_M": 72361, "RD_M": 27195},
    {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Year": 2024, "Revenue_M": 245122, "NetIncome_M": 88136, "RD_M": 29510},

    {"Ticker": "GOOGL", "Company": "Alphabet Inc.", "Year": 2021, "Revenue_M": 257637, "NetIncome_M": 76033, "RD_M": 31562},
    {"Ticker": "GOOGL", "Company": "Alphabet Inc.", "Year": 2022, "Revenue_M": 282836, "NetIncome_M": 59972, "RD_M": 39500},
    {"Ticker": "GOOGL", "Company": "Alphabet Inc.", "Year": 2023, "Revenue_M": 307394, "NetIncome_M": 73795, "RD_M": 45427},
    {"Ticker": "GOOGL", "Company": "Alphabet Inc.", "Year": 2024, "Revenue_M": 350014, "NetIncome_M": 94254, "RD_M": 48200},

    {"Ticker": "NVDA", "Company": "NVIDIA Corp.", "Year": 2021, "Revenue_M": 16675, "NetIncome_M": 4332, "RD_M": 3924},
    {"Ticker": "NVDA", "Company": "NVIDIA Corp.", "Year": 2022, "Revenue_M": 26914, "NetIncome_M": 9752, "RD_M": 5268},
    {"Ticker": "NVDA", "Company": "NVIDIA Corp.", "Year": 2023, "Revenue_M": 26974, "NetIncome_M": 4368, "RD_M": 7339},
    {"Ticker": "NVDA", "Company": "NVIDIA Corp.", "Year": 2024, "Revenue_M": 60922, "NetIncome_M": 29760, "RD_M": 8675},
]
df_fin = pd.DataFrame(FINANCIAL_DATA)
df_fin["NetMargin_%"] = (df_fin["NetIncome_M"] / df_fin["Revenue_M"] * 100).round(1)
df_fin["RD_Pct_Revenue"] = (df_fin["RD_M"] / df_fin["Revenue_M"] * 100).round(1)

# ─── Cache Agent Initialization ───────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Agentic RAG System (Loading BM25 & Embeddings)...")
def get_agent():
    return AgenticRAG(use_reranker=True)

agent = get_agent()
scope_validator = FinancialScopeValidator()

# ─── Session State Chat History ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Corpus Details")
    st.caption("• **11,127 Text Chunks**")
    st.caption("• **10 S&P 500 Tech & Financial Giants**")
    st.caption("• **Years Covered**: 2016 – 2025")
    st.caption("• **Retrieval**: Hybrid (Dense ChromaDB + BM25 RRF)")
    st.caption("• **Reranker**: Cross-Encoder MS-MARCO")

# ─── Render Existing Chat History ─────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Cited Sources ({len(message['sources'])} chunks)"):
                for idx, src in enumerate(message["sources"], 1):
                    st.caption(f"**Chunk #{idx}** — [{src.get('ticker')}/{src.get('fiscal_year')}/{src.get('section')}]")

# ─── Chat Input ───────────────────────────────────────────────────────
user_query = st.chat_input("Ask a multi-turn financial question (e.g., What was Apple's 2024 net income?)...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process assistant response
    with st.chat_message("assistant"):
        ticker_filter = None if ticker_option == "All" else ticker_option
        year_filter = None if year_option == "All" else int(year_option)
        agent.max_retries = max_retries

        start_time = time.time()
        with st.spinner("Agent searching SEC filings with conversation memory..."):
            result = agent.ask(
                query=user_query,
                ticker=ticker_filter,
                fiscal_year=year_filter,
                chat_history=st.session_state.messages[:-1],  # Past turns
            )
        elapsed = time.time() - start_time

        answer_text = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])

        st.markdown(answer_text)
        st.caption(f"⏱️ **Execution Time**: {elapsed:.2f}s | 🔄 **Retries**: {result.get('retries', 0)}")

        if sources:
            with st.expander(f"📚 Cited Sources ({len(sources)} chunks)"):
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    t = src.get("ticker") or meta.get("ticker", "N/A")
                    fy = src.get("fiscal_year") or meta.get("fiscal_year", "N/A")
                    sec = src.get("section") or meta.get("section", "N/A")
                    st.markdown(f"**Chunk #{idx}** — `[{t}/{fy}/{sec}]`")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
        })

# ─── Financial Ratios & Visual Analytics Section ────────────────────
st.markdown("---")
st.header("📊 Interactive Financial Analytics & Multi-Year Trends")

tab_charts, tab_ratios, tab_compare = st.tabs([
    "📈 Multi-Year Revenue & Net Income Trends",
    "🧮 Financial Ratio Performance",
    "⚔️ Side-by-Side Company Analytics",
])

with tab_charts:
    st.subheader("Revenue & Net Income Comparison (2021 – 2024)")
    col_metric, col_year = st.columns(2)
    with col_metric:
        selected_metric = st.selectbox(
            "Select Metric to Visualize:",
            options=["Revenue_M", "NetIncome_M", "RD_M"],
            format_func=lambda x: {"Revenue_M": "Total Revenue ($M)", "NetIncome_M": "Net Income ($M)", "RD_M": "R&D Spend ($M)"}[x],
        )
    with col_year:
        selected_company_filter = st.multiselect(
            "Select Companies:",
            options=df_fin["Ticker"].unique(),
            default=["AAPL", "MSFT", "GOOGL", "NVDA"],
        )

    filtered_df = df_fin[df_fin["Ticker"].isin(selected_company_filter)]

    fig = px.bar(
        filtered_df,
        x="Year",
        y=selected_metric,
        color="Company",
        barmode="group",
        title=f"Multi-Year {selected_metric.replace('_M', ' ($M)')} Trend",
        labels={selected_metric: "Amount ($ Millions)", "Year": "Fiscal Year"},
        text_auto=".2s",
    )
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab_ratios:
    st.subheader("Key Financial Ratios (FY2024 Snapshot)")
    
    df_2024 = df_fin[df_fin["Year"] == 2024]
    
    cols = st.columns(len(df_2024))
    for idx, (_, row) in enumerate(df_2024.iterrows()):
        with cols[idx]:
            st.metric(
                label=f"🍎 {row['Ticker']} (2024)",
                value=f"${row['NetIncome_M']:,}M",
                delta=f"Net Margin: {row['NetMargin_%']}%",
            )
            st.caption(f"R&D / Revenue: **{row['RD_Pct_Revenue']}%**")

    fig_ratios = go.Figure()
    fig_ratios.add_trace(go.Bar(
        x=df_2024["Ticker"],
        y=df_2024["NetMargin_%"],
        name="Net Profit Margin (%)",
        marker_color="#00CC96",
    ))
    fig_ratios.add_trace(go.Bar(
        x=df_2024["Ticker"],
        y=df_2024["RD_Pct_Revenue"],
        name="R&D % of Revenue",
        marker_color="#AB63FA",
    ))
    fig_ratios.update_layout(
        title="FY2024 Net Margin vs R&D Intensity (%)",
        barmode="group",
        template="plotly_dark",
        height=400,
        yaxis_title="Percentage (%)",
    )
    st.plotly_chart(fig_ratios, use_container_width=True)

with tab_compare:
    st.subheader("⚔️ Head-to-Head Company Comparison")
    c1, c2 = st.columns(2)
    with c1:
        comp1 = st.selectbox("Select Company 1:", options=df_fin["Ticker"].unique(), index=0)
    with c2:
        comp2 = st.selectbox("Select Company 2:", options=df_fin["Ticker"].unique(), index=1)

    df_c1 = df_fin[df_fin["Ticker"] == comp1]
    df_c2 = df_fin[df_fin["Ticker"] == comp2]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df_c1["Year"], y=df_c1["Revenue_M"], mode="lines+markers", name=f"{comp1} Revenue"))
    fig_comp.add_trace(go.Scatter(x=df_c2["Year"], y=df_c2["Revenue_M"], mode="lines+markers", name=f"{comp2} Revenue"))
    fig_comp.add_trace(go.Scatter(x=df_c1["Year"], y=df_c1["NetIncome_M"], mode="lines+markers", name=f"{comp1} Net Income", line=dict(dash="dash")))
    fig_comp.add_trace(go.Scatter(x=df_c2["Year"], y=df_c2["NetIncome_M"], mode="lines+markers", name=f"{comp2} Net Income", line=dict(dash="dash")))
    fig_comp.update_layout(
        title=f"{comp1} vs {comp2}: Multi-Year Trajectory ($ Millions)",
        template="plotly_dark",
        height=450,
        xaxis_title="Fiscal Year",
        yaxis_title="$ Millions",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.caption("Agentic Financial RAG System | Built with Streamlit, Plotly, ChromaDB, BM25, SentenceTransformers, and OpenRouter LLM")
