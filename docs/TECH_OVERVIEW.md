# RAG QA System — Technical Overview

> **What it does:** Accepts PDF uploads and answers natural-language questions about them.
> Text is embedded locally, stored in a vector database, and retrieved at query time to
> ground a Claude LLM response with inline citations.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | Chat UI + PDF upload sidebar + source panel |
| Backend | FastAPI + Uvicorn | REST API (`/ingest`, `/query`, `/health`) |
| PDF parsing | LangChain `PyPDFLoader` | Extracts text + page numbers from PDFs |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits pages into 500-char chunks (50-char overlap) |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Local 384-dim vectors — no external API, no cost |
| Vector DB | ChromaDB (HNSW, cosine) | Stores + searches chunk vectors on local disk |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) | Generates cited answers from retrieved context |
| LLM framework | LangChain (`ChatAnthropic` + `ChatPromptTemplate`) | Chains prompt → LLM, handles message format |
| Containerisation | Docker Compose | Two containers, named volumes, health-check ordering |
| CI/CD | GitHub Actions + GHCR | Semver gate → git tag → Docker build → registry push |
| Testing | pytest + FastAPI TestClient | Unit (mocked) + integration layers |
| Evaluation | Ragas (faithfulness, precision, recall) + keyword match | Measures retrieval and answer quality |

---

## Architecture

```
Browser
  │
  │ HTTP :8501
  ▼
Streamlit (frontend/app.py)
  │
  │ POST /ingest    POST /query
  │ HTTP :8000
  ▼
FastAPI (backend/main.py)
  ├── ingestion.py ──→ ChromaDB (./chroma_db)
  └── retrieval.py ──→ ChromaDB + Claude API
```

**Two processes, one shared database on disk.**
The frontend never touches ChromaDB directly — it speaks HTTP to the backend only.

---

## Data Flow: Ingestion (`POST /ingest`)

```
1. Streamlit uploads PDF as multipart/form-data
2. FastAPI saves to ./uploads/, enforces 50 MB limit
3. PyPDFLoader extracts pages → list of LangChain Documents
4. RecursiveCharacterTextSplitter → ~500-char chunks
5. SentenceTransformer.encode() → 384-dim float vectors
6. ChromaDB.upsert()
     ids        → md5(source:chunk_index:text[:80])   ← deterministic, re-ingest safe
     embeddings → float vectors
     documents  → raw chunk text
     metadatas  → { source, page, chunk_index }
7. Return { filename, chunks_ingested }
```

---

## Data Flow: Query (`POST /query`)

```
1. Streamlit POSTs { "question": "..." }
2. FastAPI validates (Pydantic) → calls answer_query()
3. Same SentenceTransformer embeds the question → query vector
4. ChromaDB.query() cosine search → top-10 chunks + distances
5. distance converted to similarity: score = 1 − distance
6. format_context() → "[source.pdf, page N]\n<text>\n---\n..."
7. ChatPromptTemplate fills { context, question }
8. LangChain chain: _prompt | ChatAnthropic → LLM response
9. Return { answer, sources[] }
     sources include: source, page, chunk_index, similarity_score, excerpt (200 chars)
```

---

## Key Design Decisions

| Decision | What & Why |
|---|---|
| **Local embeddings** | `all-MiniLM-L6-v2` runs on device — no API key, no per-call cost, no privacy risk for sensitive PDFs. Trade-off: ~80 MB model in the container image. |
| **Singleton embedder + ChromaDB client** | Both loaded once on server start, shared across ingestion and retrieval. Re-loading per request would cost ~1s and ~300 MB every call. |
| **Same model, both sides** | Ingestion and retrieval both call `get_embedder()` from `ingestion.py`. If they diverged, similarity scores would be meaningless. |
| **Deterministic chunk IDs (MD5)** | `md5(source:index:text[:80])` → `upsert` overwrites safely. Re-ingesting the same PDF never creates duplicates. |
| **ChromaDB stores cosine distances** | `score = 1 − distance` on the way out. High score = more relevant. |
| **LLM per-request, not singleton** | `_get_llm()` creates a new `ChatAnthropic` each call. It holds no state; model/timeout can change without restart. |
| **Temperature 0.0** | Deterministic, factual answers — correct for a document Q&A use case. |
| **"Answer only from context" prompt** | Prevents the LLM from mixing in training knowledge and generating uncited claims. |
| **`async def ingest`, `def query`** | File upload is I/O-bound → `async`. Query calls synchronous libs (ChromaDB, SentenceTransformers) → FastAPI runs it in a thread pool. |

---

## API Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/ingest` | `multipart/form-data` — field `file` (PDF ≤ 50 MB) | `{"filename": "...", "chunks_ingested": N}` |
| `POST` | `/query` | `{"question": "..."}` | `{"answer": "...", "sources": [...]}` |

Error codes: `400` bad input, `413` file too large, `500` ingestion/LLM failure.

---

## Configuration (`backend/config.py`)

All values read from environment variables with safe defaults.
**Never read `os.environ` directly elsewhere** — only `config.py`.

| Key | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for LLM calls |
| `CHROMA_DB_PATH` | `./chroma_db` | Persistent vector storage location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Change both ingestion AND retrieval if swapped |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model ID passed to ChatAnthropic |
| `LLM_TEMPERATURE` | `0.0` | Deterministic mode |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Shared chars at chunk boundaries |
| `TOP_K_RESULTS` | `10` | Chunks retrieved per query |
| `MAX_FILE_SIZE_MB` | `50` | PDF upload limit |

---

## Testing

```
tests/
  unit/
    test_ingestion.py    ← _doc_id, singleton behaviour, ingest_pdf branches
    test_retrieval.py    ← format_context, retrieve_chunks, answer_query paths
  integration/
    test_api.py          ← all endpoints via FastAPI TestClient
```

- **Unit tests** mock everything: ChromaDB, SentenceTransformer, LangChain, LLM.
- **Integration tests** mock only `ingest_pdf` and `answer_query` — test routing, validation, error handling.
- No real API calls. Run with `ANTHROPIC_API_KEY=dummy make test`.
- Coverage threshold: 70% minimum (enforced in CI).

---

## CI/CD Pipeline (3 layers)

```
[1] git commit → pre-commit hooks
      Black (auto-formats), Flake8, trailing-whitespace, check-yaml, large-file guard

[2] git push → pre-push hook
      scripts/check_version_bump.py
      reads origin/main:backend/__version__.py — aborts if local version ≤ remote

[3] GitHub Actions (cd.yml → _prep.yml → _build-push.yml)
      _prep.yml    : extract version from __version__.py
                     validate semver > latest git tag (can't be bypassed with --no-verify)
                     create annotated git tag vX.Y.Z on most-recent non-merge commit
      _build-push.yml : login to GHCR with GITHUB_TOKEN (no stored secrets)
                        build backend + frontend Docker images
                        push 3 tags: :X.Y.Z  :sha-<short>  :latest
                        registry-based layer cache (avoids GH Actions cache quota)
```

**Single source of truth for the version:** `backend/__version__.py`.
Everything — `make version`, the pre-push hook, the CI tag, the Docker image tag — reads from this file.

---

## Evaluation

```bash
make evals        # keyword match only — fast, no API calls
make evals-full   # adds Ragas metrics (calls Claude as judge)
```

20 Q&A pairs written for "Attention Is All You Need" (Vaswani et al., 2017).

**Metrics:**

| Metric | What it checks |
|---|---|
| **Keyword hit** | Answer contains expected keywords (free, fast) |
| **Hallucination flag** | Retrieval succeeded but answer missed keywords → possible confabulation |
| **Context Precision** (Ragas) | Retrieved chunks are relevant to the question |
| **Context Recall** (Ragas) | All necessary information was retrieved |
| **Answer Faithfulness** (Ragas) | Answer is grounded in the retrieved context, not training knowledge |

---

## Docker Setup

```yaml
services:
  backend:   port 8000, volumes: chroma_data + upload_data
  frontend:  port 8501, depends_on backend (service_healthy)

healthcheck: GET /health every 30s, start_period 30s
```

Embedding model baked into the image at build time (`RUN python -c "SentenceTransformer(...)"`) —
avoids a 60s download on first container start.

Named volumes (`chroma_data`, `upload_data`) survive container restarts; ingested documents persist.
