import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("RAG Question-Answering System")

# ── Sidebar: PDF upload ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Ingestion")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Ingesting…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    st.success(f"Ingested {data['chunks_ingested']} chunks from {data['filename']}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is the backend service running?")

    st.divider()
    st.caption("Backend: " + BACKEND_URL)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "answer": str, "sources": list}

# ── Main layout: chat left, sources right ─────────────────────────────────────
chat_col, source_col = st.columns([3, 2])

with chat_col:
    st.subheader("Chat")
    for turn in st.session_state.history:
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Assistant:** {turn['answer']}")
        st.divider()

    question = st.text_input("Ask a question about your documents", key="input")
    if st.button("Send") and question.strip():
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question},
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    st.session_state.history.append(
                        {"question": question, "answer": data["answer"], "sources": data["sources"]}
                    )
                    st.rerun()
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is the backend service running?")

with source_col:
    st.subheader("Source Chunks")
    if st.session_state.history:
        last = st.session_state.history[-1]
        if last["sources"]:
            for i, src in enumerate(last["sources"], 1):
                score = src.get("similarity_score", 0)
                with st.expander(f"[{i}] {src['source']} — page {src['page']}  (score: {score:.3f})"):
                    st.caption(f"Chunk #{src.get('chunk_index', '?')}")
                    st.write(src.get("excerpt", ""))
        else:
            st.info("No sources retrieved.")
    else:
        st.info("Sources will appear here after your first query.")
