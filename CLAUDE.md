# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

All common operations are available via `make`. Run `make help` to see the full list.

**First-time setup (after cloning)**
```bash
make install    # pip install -r requirements.txt -r requirements-dev.txt
make hooks      # install pre-commit + pre-push hooks
```

**Run the project locally**
```bash
# Terminal 1
ANTHROPIC_API_KEY=sk-ant-... make backend    # FastAPI on :8000

# Terminal 2
make frontend                                # Streamlit on :8501
```

**Docker (full stack)**
```bash
ANTHROPIC_API_KEY=sk-ant-... make up        # build + start both services
make down                                   # stop
```

**Tests**
```bash
make test             # full suite + coverage (min 70%)
make test-unit        # unit tests only (fast, no deps)
make test-integration # integration tests only
```

**Code quality**
```bash
make format   # auto-format with Black
make lint     # Flake8
make check    # black --check + flake8 (same as CD pipeline)
```

**Evals (backend must be running)**
```bash
make evals        # keyword-match only, no API calls
make evals-full   # full Ragas metrics (requires ANTHROPIC_API_KEY)
```

**Other**
```bash
make version  # print current __version__
make clean    # remove __pycache__, .pytest_cache, coverage files
```

Raw commands (when make is not available):
```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && streamlit run app.py

# Tests
pytest tests/ -v --cov=backend --cov-report=term-missing

# Docker
docker compose up --build
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

## Testing

Tests live in `tests/` and are split into two layers:

```
tests/
  unit/
    test_ingestion.py   ← _doc_id, embedder/collection singletons, ingest_pdf branches
    test_retrieval.py   ← format_context, retrieve_chunks, answer_query all paths
  integration/
    test_api.py         ← all API endpoints via FastAPI TestClient
```

- Unit tests mock all external dependencies (ChromaDB, SentenceTransformer, LangChain, LLM).
- Integration tests mock only `ingest_pdf` and `answer_query` — they test the API routing, validation, and error handling against the real FastAPI app.
- `pythonpath = ["backend"]` in `pyproject.toml` adds `backend/` to sys.path automatically — no `conftest.py` path hacks needed.
- Run `ANTHROPIC_API_KEY=dummy make test` when the API key is not set — tests do not make real API calls.

## Code Quality

Config is centralised in `pyproject.toml` (`[tool.black]`, `[tool.flake8]`, `[tool.pytest.ini_options]`).

**Pre-commit hooks** (run on every `git commit`):
- trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files
- Black auto-formats Python files
- Flake8 lints (requires `flake8-pyproject` to read `pyproject.toml`)
- `pip-audit` scans `requirements.txt` for known CVEs

**Pre-push hook** (runs on `git push`):
- `scripts/check_version_bump.py` — compares `backend/__version__.py` against the latest git tag and fails if the version was not bumped.

Install hooks once after cloning: `make hooks`

## Versioning

The project version lives in `backend/__version__.py`:
```python
__version__ = "0.1.0"
```

Bump this file before every push to `main`. The pre-push hook will reject the push if the version hasn't changed. The CD pipeline uses this value to create an annotated git tag and to tag Docker images.

## CI/CD

**CD pipeline** (`.github/workflows/`):

Triggers on every push to `main` (and manually via `workflow_dispatch`).

```
cd.yml  →  _prep.yml              →  _build-push.yml
           • extract version          • login to GHCR
           • validate semver bump     • build backend image
           • create annotated tag     • build frontend image
                                      • push both (SHA + version + latest tags)
                                      • Trivy CVE scan (CRITICAL/HIGH, --ignore-unfixed)
                                      • upload SARIF to GitHub Security tab
```

Images are pushed to `ghcr.io/<owner>/rag-qa-system-{backend,frontend}`.
No extra secrets are needed — the CD uses `GITHUB_TOKEN` for GHCR authentication.

## Environment Variables

| Variable | Required | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — |
| `CHROMA_DB_PATH` | No | `./chroma_db` |

## Eval Pairs

`evals/eval_pairs.py` contains 20 Q&A pairs written for **"Attention Is All You Need" (Vaswani et al., 2017)**. Each pair has `expected_keywords` (a list of strings) used for keyword-hit detection — not exact-match. If you change the target document, replace the pairs and keywords in that file; the eval runner in `run_evals.py` is document-agnostic.

## Security Constraints

- Never include actual API keys, tokens, or credentials in examples
- Use placeholder values like `YOUR_API_KEY_HERE` in configuration samples
- Never demonstrate permission configurations that bypass security controls

## Claude Code Plugin Setup

This project uses Claude Code plugins defined in `.claude/settings.json`. Plugins are enabled per-project but must be **installed once per machine**. After cloning, run:

```bash
for p in superpowers context7 code-simplifier skill-creator claude-code-setup security-guidance; do
  claude plugin install "$p@claude-plugins-official"
done
```

## Key Claude Behaviour

- Don't always agree with me. Be blunt and always propose the best solution in terms of complexity, maintainability and performance.
- Be direct and correct me if I make wrong assumptions or give you wrong information.
