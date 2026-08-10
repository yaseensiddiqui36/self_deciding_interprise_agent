"""Streamlit UI for the self-correcting enterprise data agent.

Run standalone (no separate backend needed):
    streamlit run src/infinite_coding_round/ui/streamlit_app.py

Or point it at a running FastAPI backend:
    AGENT_MODE=api AGENT_API_URL=http://127.0.0.1:8000 streamlit run ...

By default (AGENT_MODE=auto) it tries the FastAPI backend first and transparently
falls back to calling the agent in-process if no backend is reachable -- this is what
lets the same file work both in local dev (decoupled API) and on a single-service
deployment (e.g. Streamlit Community Cloud, where no separate backend is running).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `infinite_coding_round` importable even when this file is run directly by
# Streamlit Cloud without the project having been pip-installed first.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

# Secrets configured in Streamlit Cloud (Settings -> Secrets) are only exposed via
# st.secrets, not as real environment variables. Promote them to env vars *before*
# importing anything from infinite_coding_round, since Settings() reads env at import time.
for _key in ("GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL"):
    if _key not in os.environ:
        try:
            if _key in st.secrets:
                os.environ[_key] = str(st.secrets[_key])
        except Exception:
            pass

import requests  # noqa: E402

API_BASE_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000")
AGENT_MODE = os.getenv("AGENT_MODE", "auto")  # "api" | "direct" | "auto"

EXAMPLE_QUESTIONS = [
    "How many orders does customer Alice Chen have?",
    "What is the refund window for furniture items?",
    "Hassan Ali reported a noisy standing desk motor after 40 days. Is he eligible for a refund, and should this be escalated?",
    "What is the CEO's stance on remote work?",
]

st.set_page_config(
    page_title="Enterprise Data Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1000px; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.08);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        background: rgba(99, 102, 241, 0.15);
        color: rgb(99, 102, 241);
    }
    .source-chip {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        background: rgba(16, 185, 129, 0.12);
        color: rgb(5, 150, 105);
        margin: 0 0.25rem 0.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Agent access: HTTP (decoupled FastAPI backend) with in-process fallback
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _get_direct_agent():
    """Lazily builds the LangGraph agent in-process (used for standalone deployment)."""
    from infinite_coding_round.agent.graph import run_agent
    from infinite_coding_round.db.seed import seed_database
    from infinite_coding_round.config import DB_PATH
    from infinite_coding_round.rag.vectorstore import load_or_build_index

    if not DB_PATH.exists():
        seed_database()
    load_or_build_index()  # builds the FAISS index on first run if missing
    return run_agent


def _call_via_api(question: str) -> dict:
    response = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


def _call_direct(question: str) -> dict:
    run_agent = _get_direct_agent()
    state = run_agent(question)
    return {
        "answer": state.get("answer", ""),
        "tools_used": state.get("tools_used", []),
        "generated_sql": state.get("sql_query"),
        "retrieved_sources": state.get("retrieved_sources", []),
        "confidence": state.get("confidence", 0.0),
        "validation_result": state.get("validation_reason", ""),
        "validation_passed": state.get("validation_passed", False),
        "retry_count": state.get("retry_count", 0),
    }


def ask_agent(question: str) -> tuple[dict, str]:
    """Returns (result, mode_used)."""
    if AGENT_MODE == "direct":
        return _call_direct(question), "direct"
    if AGENT_MODE == "api":
        return _call_via_api(question), "api"

    # auto: prefer API, remember the fallback for the rest of the session
    if not st.session_state.get("force_direct_mode"):
        try:
            return _call_via_api(question), "api"
        except requests.RequestException:
            st.session_state["force_direct_mode"] = True
    return _call_direct(question), "direct"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### 📊 Enterprise Data Agent")
    st.caption(
        "Routes each question to SQL, document retrieval, or both — then validates "
        "its own grounding and retries once automatically on failure."
    )

    st.markdown("#### Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.divider()
    st.markdown("#### Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    st.divider()
    mode_label = {
        "api": f"FastAPI backend ({API_BASE_URL})",
        "direct": "In-process (standalone)",
        "auto": "Auto (API, falls back to standalone)",
    }[AGENT_MODE]
    st.caption(f"Mode: **{mode_label}**")
    if st.session_state.get("force_direct_mode"):
        st.caption("⚠️ Backend unreachable — running standalone this session.")


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #

st.title("Self-Correcting Enterprise Data Agent")
st.caption("Ask a business question about customers, orders, refunds, or escalation policy.")

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["content"])
            continue

        result = turn["result"]
        st.write(result["answer"])

        badges = " ".join(f'<span class="agent-badge">{t}</span>' for t in result["tools_used"])
        if badges:
            st.markdown(badges, unsafe_allow_html=True)

        cols = st.columns(3)
        cols[0].metric("Confidence", f"{result['confidence']:.2f}")
        cols[1].metric("Retries", result["retry_count"])
        cols[2].metric("Validation", "✅ Passed" if result["validation_passed"] else "❌ Failed")

        if result.get("generated_sql"):
            with st.expander("Generated SQL"):
                st.code(result["generated_sql"], language="sql")

        if result.get("retrieved_sources"):
            with st.expander(f"Retrieved sources ({len(result['retrieved_sources'])})"):
                for src in result["retrieved_sources"]:
                    st.markdown(f'<span class="source-chip">{src["source"]}</span>', unsafe_allow_html=True)
                    st.caption(src["excerpt"])
                    st.divider()

        with st.expander("Validation reasoning"):
            st.write(result["validation_result"])


pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask about orders, customers, refunds, escalation...")
question = pending or typed

if question:
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result, mode_used = ask_agent(question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"The agent failed to answer: {exc}")
                st.stop()

        st.write(result["answer"])
        badges = " ".join(f'<span class="agent-badge">{t}</span>' for t in result["tools_used"])
        if badges:
            st.markdown(badges, unsafe_allow_html=True)

        cols = st.columns(3)
        cols[0].metric("Confidence", f"{result['confidence']:.2f}")
        cols[1].metric("Retries", result["retry_count"])
        cols[2].metric("Validation", "✅ Passed" if result["validation_passed"] else "❌ Failed")

        if result.get("generated_sql"):
            with st.expander("Generated SQL"):
                st.code(result["generated_sql"], language="sql")

        if result.get("retrieved_sources"):
            with st.expander(f"Retrieved sources ({len(result['retrieved_sources'])})"):
                for src in result["retrieved_sources"]:
                    st.markdown(f'<span class="source-chip">{src["source"]}</span>', unsafe_allow_html=True)
                    st.caption(src["excerpt"])
                    st.divider()

        with st.expander("Validation reasoning"):
            st.write(result["validation_result"])

    st.session_state["history"].append({"role": "assistant", "result": result})
