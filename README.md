# RAG Question-Answering System

A production-ready Retrieval-Augmented Generation (RAG) pipeline that answers questions about PDF documents using local embeddings and Claude as the reasoning engine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Browser                               │
│                       (Streamlit :8501)                             │
│    ┌───────────────┐              ┌──────────────────────────────┐  │
│    │  Chat Panel   │              │  Source Panel (cited chunks) │  │
│    └───────┬───────┘              └──────────────────────────────┘  │
└────────────┼────────────────────────────────────────────────────────┘
             │ HTTP POST /query or /ingest
             ▼
┌────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend :8000                          │
│                                                                     │
│   POST /ingest                         POST /query                  │
│   ┌────────────────────────┐           ┌───────────────────────┐    │
│   │  PyPDFLoader           │           │  Query Embedding      │    │
│   │  RecursiveTextSplitter │           │  (sentence-transformers│    │
│   │  SentenceTransformer   │           │   all-MiniLM-L6-v2)   │    │
│   │  (all-MiniLM-L6-v2)   │           └──────────┬────────────┘    │
│   └────────────┬───────────┘                      │                 │
│                │ upsert                     cosine │ search          │
│                ▼                                  ▼                 │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                   ChromaDB (persistent)                    │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                   │ top-K chunks    │
│                                                   ▼                 │
│                                    ┌──────────────────────────┐     │
│                                    │  LangChain RAG Chain     │     │
│                                    │  ChatAnthropic           │     │
│                                    │  (claude-sonnet-4-...)   │     │
│                                    └──────────────────────────┘     │
│                                                   │ answer + sources│
└───────────────────────────────────────────────────┼────────────────┘
                                                    ▼
                                           JSON response to UI
```

---

## What This Proves

This project demonstrates the following production AI/ML engineering skills:

| Skill | Where demonstrated |
|---|---|
| **RAG pipeline design** | `ingestion.py` → `retrieval.py`: chunk → embed → store → retrieve → generate |
| **Local embeddings** | `sentence-transformers` runs entirely on-device — no external embedding API, no cost, no latency |
| **LLM integration** | `langchain-anthropic` with structured prompt templates, timeout handling, and citation-grounded answers |
| **Vector database** | ChromaDB with cosine-similarity search, persistent storage, and upsert-safe document IDs |
| **REST API design** | FastAPI with Pydantic models, CORS, multipart file upload, health-check endpoint |
| **Frontend** | Streamlit two-panel UI (chat + cited sources) — usable without any frontend framework knowledge |
| **Evaluation mindset** | `evals/` folder with 20 Q&A pairs, keyword-match retrieval check, hallucination flag, and Ragas metrics |
| **Containerization** | Multi-service Docker Compose with volume persistence, health checks, and env-variable secrets |
| **Testing** | 21-test suite across unit and integration layers; all external dependencies mocked |
| **CI/CD** | 3-layer GitHub Actions CD pipeline: semver gate → Docker build → GHCR push → Trivy CVE scan |
| **Code quality** | pre-commit hooks (Black, Flake8, pip-audit), pyproject.toml central config, coverage enforcement |
| **Error handling** | Graceful fallbacks at both retrieval and LLM layers; clear error messages to the user |

---

## Quickstart (Docker)

```bash
# 1. Clone
git clone https://github.com/NNtorvas/RAG-QA-SYSTEM.git && cd RAG-QA-SYSTEM

# 2. Build and start
ANTHROPIC_API_KEY=sk-ant-... make up
# or without make:
# ANTHROPIC_API_KEY=sk-ant-... docker compose up --build

# 3. Open the UI
open http://localhost:8501
```

Upload a PDF in the sidebar, wait for ingestion, then ask questions.

---

## Quickstart (local dev)

```bash
# 1. Install dependencies and hooks
make install
make hooks

# 2. Terminal 1 — backend
ANTHROPIC_API_KEY=sk-ant-... make backend

# 3. Terminal 2 — frontend
make frontend
```

---

## Testing

```bash
make test             # full suite with coverage report (min 70%)
make test-unit        # unit tests only — fast, no external deps
make test-integration # API-level tests via FastAPI TestClient
```

Tests do not call the Anthropic API or load the embedding model. All external services are mocked.

---

## Code Quality

```bash
make format   # auto-format with Black
make check    # black --check + flake8 (same checks as CD pipeline)
```

Pre-commit hooks run automatically on `git commit` (formatting, linting, CVE scan) and on `git push` (version bump check). Install them once with `make hooks`.

---

## Running Evaluations

The eval script targets "Attention Is All You Need" (Vaswani et al., 2017). Ingest the paper first, then:

```bash
make evals        # keyword-only (fast, no extra API calls)
make evals-full   # full Ragas metrics (requires ANTHROPIC_API_KEY + backend running)
```

### Example output

```
## Eval Results

| ID   | Question (truncated)                                           | Retrieval | KW Match | Hallucination | Result |
|------|----------------------------------------------------------------|-----------|----------|---------------|--------|
| q01  | What is the main architecture proposed in the paper?          | yes       | yes      | no            | PASS   |
| q02  | What problem does self-attention solve compared to RNNs?      | yes       | yes      | no            | PASS   |
| q03  | How many attention heads are used in the base model?          | yes       | yes      | no            | PASS   |
...

**Summary:** 18/20 passed

## Ragas Scores

| Metric               | Score  |
|----------------------|--------|
| context_precision    | 0.8750 |
| context_recall       | 0.8100 |
| answer_faithfulness  | 0.9200 |
```

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}`. Used by Docker health check.

### `POST /ingest`
Upload a PDF for processing.

**Request:** `multipart/form-data` with field `file` (PDF, max 50 MB)

**Response:**
```json
{"filename": "paper.pdf", "chunks_ingested": 142}
```

### `POST /query`
Ask a question about ingested documents.

**Request:**
```json
{"question": "What optimizer is used?"}
```

**Response:**
```json
{
  "answer": "The model uses the Adam optimizer [paper.pdf, page 7]...",
  "sources": [
    {
      "source": "paper.pdf",
      "page": "7",
      "chunk_index": "83",
      "similarity_score": 0.8921,
      "excerpt": "We used the Adam optimizer with β1 = 0.9..."
    }
  ]
}
```

---

## Configuration

All configuration lives in `backend/config.py` and is overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `CHROMA_DB_PATH` | `./chroma_db` | Where ChromaDB persists data |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |

---

## Project Structure

```
rag-qa-system/
├── backend/
│   ├── main.py             # FastAPI routes (/ingest, /query, /health)
│   ├── ingestion.py        # PDF → chunks → embeddings → ChromaDB
│   ├── retrieval.py        # ChromaDB search + LangChain RAG chain
│   ├── config.py           # Settings from env vars
│   └── __version__.py      # Single source of truth for project version
├── frontend/
│   └── app.py              # Streamlit UI (chat + source panel)
├── tests/
│   ├── unit/
│   │   ├── test_ingestion.py   # _doc_id, singletons, ingest_pdf branches
│   │   └── test_retrieval.py   # format_context, retrieve_chunks, answer_query
│   └── integration/
│       └── test_api.py         # all endpoints via FastAPI TestClient
├── evals/
│   ├── eval_pairs.py       # 20 hardcoded Q&A pairs
│   └── run_evals.py        # Ragas eval runner
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── scripts/
│   └── check_version_bump.py   # pre-push hook: validates semver increment
├── .github/
│   └── workflows/
│       ├── cd.yml              # CD entry point (push to main)
│       ├── _prep.yml           # reusable: version validation + git tag
│       └── _build-push.yml     # reusable: Docker build + GHCR push + Trivy
├── Makefile                # developer task runner (make help)
├── pyproject.toml          # Black + Flake8 + pytest config
├── .pre-commit-config.yaml # commit/push quality gates
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # dev/test dependencies
└── docker-compose.yml
```
