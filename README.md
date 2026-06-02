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

> **For recruiters and hiring managers**

This project demonstrates the following production ML engineering skills:

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
| **Error handling** | Graceful fallbacks at both retrieval and LLM layers; clear error messages to the user |

---

## Quickstart (Docker)

```bash
# 1. Clone and enter the project
git clone https://github.com/NNtorvas/RAG-QA-SYSTEM.git && cd RAG-QA-SYSTEM

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Build and start
docker compose up --build

# 4. Open the UI
open http://localhost:8501
```

Upload a PDF in the sidebar, wait for ingestion, then ask questions.

---

## Quickstart (local dev)

```bash
# Python 3.11+ recommended
pip install -r requirements.txt

# Terminal 1 — backend
cd backend
ANTHROPIC_API_KEY=sk-ant-... uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

---

## Running Evaluations

The eval script targets "Attention Is All You Need" (Vaswani et al., 2017). Ingest the paper first, then:

```bash
# Keyword-only (fast, no extra API calls)
python evals/run_evals.py --backend http://localhost:8000 --skip-ragas

# Full Ragas metrics (context precision/recall + faithfulness)
ANTHROPIC_API_KEY=sk-ant-... python evals/run_evals.py --backend http://localhost:8000
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
│   ├── main.py         # FastAPI routes (/ingest, /query, /health)
│   ├── ingestion.py    # PDF → chunks → embeddings → ChromaDB
│   ├── retrieval.py    # ChromaDB search + LangChain RAG chain
│   └── config.py       # Settings from env vars
├── frontend/
│   └── app.py          # Streamlit UI (chat + source panel)
├── evals/
│   ├── eval_pairs.py   # 20 hardcoded Q&A pairs
│   └── run_evals.py    # Ragas eval runner
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```
