"""Streamlit frontend for the IoT Department Assistant.

A chat UI that talks to the FastAPI backend.  Features:
  - chat history
  - sidebar dropdowns for year (1–4 or All) and semester (1, 2, or All)
  - a "Sources" expander under each answer
  - a debug toggle showing retrieved chunks and scores
  - a re-index button that calls POST /ingest

Run the backend first::

    uvicorn app.api:app --reload

Then run this app::

    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

# Backend URL (override with API_URL env var if needed)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 600  # seconds; local LLMs can be slow


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def get_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def get_documents() -> list[dict]:
    try:
        return requests.get(f"{API_URL}/documents", timeout=10).json()
    except Exception:  # noqa: BLE001
        return []


def ask_backend(question: str, year, semester) -> dict:
    payload: dict = {"question": question}
    if year is not None:
        payload["year"] = year
    if semester is not None:
        payload["semester"] = semester
    resp = requests.post(f"{API_URL}/ask", json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.text
        raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")
    return resp.json()


def trigger_reindex() -> dict:
    resp = requests.post(f"{API_URL}/ingest", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="IoT Department Assistant", page_icon="🎓", layout="centered")

st.title("🎓 IoT Department Assistant")
st.caption("Ask about the IoT syllabus (all 4 years) and the academic calendar.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content, sources, debug}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    year_choice = st.selectbox("Year of study", ["All", "1", "2", "3", "4"], index=0)
    sem_choice = st.selectbox("Semester", ["All", "1", "2"], index=0)

    year = None if year_choice == "All" else int(year_choice)
    semester = None if sem_choice == "All" else int(sem_choice)

    st.divider()
    show_debug = st.toggle("Show debug info", value=False)

    st.divider()
    st.header("Status")
    health = get_health()
    if health.get("status") == "ok":
        st.success("Backend: connected")
        st.caption(f"Ollama: {health.get('ollama')}")
        st.caption(f"Model: {health.get('llm_model')}")
        st.caption(f"Chunks indexed: {health.get('chunks_indexed')}")
    else:
        st.error("Backend not reachable")
        st.caption(f"Expected at {API_URL}")
        st.caption(str(health.get("error", "")))

    with st.expander("Indexed documents"):
        docs = get_documents()
        if docs:
            for d in docs:
                st.caption(f"{d['chunk_count']:>4} · [{d['doc_type']}] {d['source_file']}")
        else:
            st.caption("No documents indexed yet.")

    st.divider()
    if st.button("🔄 Re-index documents", use_container_width=True):
        with st.spinner("Re-indexing… this can take several minutes."):
            try:
                summary = trigger_reindex()
                st.success(f"Indexed {summary.get('total_chunks', 0)} chunks.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Re-index failed: {exc}")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            code = f" · {s['course_code']}" if s.get("course_code") else ""
            event = f" · {s['event_type']}" if s.get("event_type") else ""
            st.markdown(
                f"**[{s['doc_type']}]** {s['source_file']}, p.{s['page']}{code}{event}"
            )
            st.caption(s["snippet"])


def render_debug(debug: dict) -> None:
    if not debug:
        return
    with st.expander("Debug: retrieval"):
        st.json(debug)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            if show_debug:
                render_debug(msg.get("debug", {}))


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about the IoT syllabus or calendar…")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                data = ask_backend(prompt, year, semester)
                answer = data.get("answer", "(no answer)")
                sources = data.get("sources", [])
                debug = data.get("debug", {})
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ {exc}"
                sources = []
                debug = {}

        st.markdown(answer)
        render_sources(sources)
        if show_debug:
            render_debug(debug)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "debug": debug}
    )
