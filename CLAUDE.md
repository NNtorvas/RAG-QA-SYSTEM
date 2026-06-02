# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Backend (run from `backend/`)**
```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn main:app --reload --port 8000
```

**Frontend (run from `frontend/`)**
```bash
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

**Docker (run from repo root)**
```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

**Evals (run from repo root, backend must be running)**
```bash
# Fast — keyword match only, no API calls
python evals/run_evals.py --backend http://localhost:8000 --skip-ragas

# Full — includes Ragas context precision/recall + faithfulness
ANTHROPIC_API_KEY=sk-ant-... python evals/run_evals.py --backend http://localhost:8000
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

## Architecture

The system has two services and a shared data layer:

```
frontend/app.py  →  POST /ingest, POST /query  →  backend/main.py
                                                        │
                                          ┌─────────────┴─────────────┐
                                     ingestion.py               retrieval.py
                                          │                           │
                                     ChromaDB  ←───────────── cosine search
```

**Data flow — ingestion:**
`main.py` receives the PDF upload → saves to `uploads/` → calls `ingestion.ingest_pdf()` → `PyPDFLoader` parses pages → `RecursiveCharacterTextSplitter` chunks text → `SentenceTransformer` encodes chunks locally → ChromaDB `upsert` with deterministic MD5 chunk IDs (safe to re-ingest the same PDF).

**Data flow — query:**
`main.py` receives the question → calls `retrieval.answer_query()` → embeds question with the same `SentenceTransformer` → cosine search over ChromaDB → top-K chunks formatted as `[source.pdf, page N]\n<text>` context → LangChain `ChatPromptTemplate | ChatAnthropic` chain → returns `{answer, sources}`.

## Key Design Constraints

- **Embeddings are local** (`all-MiniLM-L6-v2` via `sentence-transformers`). The model is lazy-loaded as a module-level singleton in `ingestion.py` (`_embedder`). Both ingestion and retrieval call `get_embedder()` — they must always use the same model or similarity scores break.
- **ChromaDB client is also a singleton** (`_chroma_client` in `ingestion.py`). `retrieval.py` imports `get_collection` from `ingestion.py` — there is intentionally one client shared across both modules.
- **All config flows through `backend/config.py`**. Never read `os.environ` directly elsewhere.
- **LLM is instantiated per-request** in `_get_llm()` — it is not a singleton, which is intentional (stateless, allows timeout/model to be changed without restart).
- **ChromaDB stores cosine distances** (not similarities). `retrieval.py` converts: `similarity = 1 - distance`.
- **Chunk IDs are deterministic**: `md5(source:chunk_index:text[:80])`. Re-ingesting the same PDF upserts rather than duplicates.

## Environment Variables

| Variable | Required | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — |
| `CHROMA_DB_PATH` | No | `./chroma_db` |

## Eval Pairs

`evals/eval_pairs.py` contains 20 Q&A pairs written for **"Attention Is All You Need" (Vaswani et al., 2017)**. Each pair has `expected_keywords` (a list of strings) used for keyword-hit detection — not exact-match. If you change the target document, replace the pairs and keywords in that file; the eval runner in `run_evals.py` is document-agnostic.
