# RAG QA System — Complete Deep Dive

This document explains every piece of the system: what it does, why it was built that way, what the
alternatives were, how the parts connect, and concrete examples you can trace through.

---

## Table of Contents

1. [What Is RAG?](#1-what-is-rag)
2. [Big-Picture Architecture](#2-big-picture-architecture)
3. [Configuration — config.py](#3-configuration--configpy)
4. [Ingestion Pipeline — ingestion.py](#4-ingestion-pipeline--ingestionpy)
   - 4.1 PDF Loading
   - 4.2 Text Chunking
   - 4.3 Embedding
   - 4.4 Vector Storage
   - 4.5 Deterministic IDs
5. [Retrieval Pipeline — retrieval.py](#5-retrieval-pipeline--retrievalpy)
   - 5.1 Query Embedding
   - 5.2 Cosine Search
   - 5.3 Context Formatting
   - 5.4 Prompt Template
   - 5.5 The LLM Call
6. [API Server — main.py](#6-api-server--mainpy)
7. [Frontend — frontend/app.py](#7-frontend--frontendapppy)
8. [Evaluation System — evals/](#8-evaluation-system--evals)
9. [Docker & Deployment](#9-docker--deployment)
10. [End-to-End Trace: One Question, Every Step](#10-end-to-end-trace-one-question-every-step)
11. [Glossary](#11-glossary)

---

## 1. What Is RAG?

**RAG = Retrieval-Augmented Generation.**

A plain LLM (like Claude) can only answer from what it learned during training. It has no knowledge
of your private PDF, last week's report, or any document you upload today.

RAG solves this by giving the LLM a "cheat sheet" at the moment of the question:

```
[Your PDF]  →  break into chunks  →  encode as numbers  →  store in a database
                                                                     ↓
[Your question]  →  encode as numbers  →  find the most similar chunks
                                                                     ↓
                 [chunks] + [question]  →  LLM  →  answer with citations
```

The LLM never needs to "remember" your document. It reads the relevant pieces fresh every single
time you ask a question.

**Why not just paste the whole PDF into the prompt?**
- A 200-page PDF might be ~150,000 words. That exceeds many LLM context limits.
- Even if it fits, the LLM charges you per token and gets slower/less accurate with huge contexts.
- RAG is selective: it only sends the 4 most relevant chunks (~2,000 words), keeping the prompt
  small, cheap, and focused.

---

## 2. Big-Picture Architecture

```
┌──────────────────────────────────┐
│  Browser / User                  │
└──────────┬───────────────────────┘
           │  HTTP  (port 8501)
┌──────────▼───────────────────────┐
│  Frontend  (Streamlit)           │
│  frontend/app.py                 │
│  - PDF upload UI                 │
│  - Chat UI                       │
│  - Source panel                  │
└──────────┬───────────────────────┘
           │  HTTP POST /ingest
           │  HTTP POST /query    (port 8000)
┌──────────▼───────────────────────┐
│  Backend  (FastAPI)              │
│  backend/main.py                 │
│                                  │
│  ┌───────────────────────────┐   │
│  │  ingestion.py             │   │
│  │  - load PDF               │   │
│  │  - split into chunks      │   │
│  │  - embed (local model)    │   │
│  │  - upsert → ChromaDB      │   │
│  └───────────────────────────┘   │
│                                  │
│  ┌───────────────────────────┐   │
│  │  retrieval.py             │   │
│  │  - embed question         │   │
│  │  - search ChromaDB        │   │
│  │  - format context         │   │
│  │  - call Claude API        │   │
│  │  - return answer+sources  │   │
│  └───────────────────────────┘   │
└──────────┬───────────────────────┘
           │
┌──────────▼───────────────────────┐
│  ChromaDB  (local disk)          │
│  ./chroma_db/                    │
│  - stores chunk text             │
│  - stores chunk vectors          │
│  - stores metadata               │
└──────────────────────────────────┘
```

**Two processes, one database on disk.**
The frontend never touches ChromaDB directly — it only talks HTTP to the backend.

---

## 3. Configuration — config.py

```python
# backend/config.py
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CHROMA_DB_PATH    = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME   = "documents"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL         = "claude-sonnet-4-20250514"
LLM_TEMPERATURE   = 0.0
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K_RESULTS     = 4
MAX_FILE_SIZE_MB  = 50
UPLOAD_DIR        = Path("./uploads")
```

**Why a single config file?**
Every module (`ingestion.py`, `retrieval.py`, `main.py`) imports from here. If you want to change
the chunk size or swap LLM models, you change one number in one place. Without this, you'd have
magic numbers scattered across files.

**Why `os.environ.get` and not hardcoding?**
Security. You never put API keys in source code — they would end up in git history for anyone to
see. The key is injected at runtime via an environment variable.

**Key values explained:**

| Setting | Value | Why this value |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Small (80 MB), fast, free, 384-dimensional vectors. Good trade-off between quality and speed. |
| `LLM_MODEL` | `claude-sonnet-4-*` | Smarter than Haiku, cheaper than Opus. Good default. |
| `LLM_TEMPERATURE` | `0.0` | Deterministic outputs. For Q&A from a document you want the same factual answer every time, not creative variation. |
| `CHUNK_SIZE` | `500` | ~3-4 sentences per chunk. Small enough that each chunk is about one idea; large enough to have context. |
| `CHUNK_OVERLAP` | `50` | Chunks share 50 characters at their boundary. This prevents a sentence that straddles two chunks from being split and losing meaning. |
| `TOP_K_RESULTS` | `4` | Send the 4 most relevant chunks to the LLM. More = larger prompt = more cost and potential noise. |

---

## 4. Ingestion Pipeline — ingestion.py

When a PDF is uploaded, this is the sequence:

```
PDF file  →  PyPDFLoader  →  raw pages
          →  RecursiveCharacterTextSplitter  →  chunks
          →  SentenceTransformer.encode()    →  vectors
          →  ChromaDB.upsert()               →  stored
```

### 4.1 PDF Loading

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(str(pdf_path))
raw_docs = loader.load()
```

`PyPDFLoader` reads a PDF and returns a list of LangChain `Document` objects, one per page:

```python
# What raw_docs looks like after loading a 10-page PDF:
[
  Document(page_content="Introduction\nThe Transformer...", metadata={"source": "paper.pdf", "page": 0}),
  Document(page_content="Attention Mechanisms\nWe define...", metadata={"source": "paper.pdf", "page": 1}),
  ...  # 10 items total
]
```

**Why PyPDFLoader?**
It is the simplest option that integrates with LangChain and handles the most common PDF formats.
It also preserves page numbers in metadata, which we later use for citations.

**Alternatives considered:**
- `pdfplumber` — better at tables and layout, but more complex setup, no LangChain integration.
- `PyMuPDF (fitz)` — faster and handles more edge cases, but requires a C library.
- `Textract (AWS)` — cloud-based OCR, handles scanned PDFs, but costs money per page and requires
  AWS credentials. Overkill for most use cases.
- `pdfminer` — lower level, requires more code to get the same result.

**Limitation:** PyPDFLoader cannot read scanned PDFs (images of text). For those you'd need OCR
(e.g. Tesseract). This system only handles text-based PDFs.

---

### 4.2 Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
```

This takes the raw pages and splits them into smaller pieces.

**Why do we chunk at all?**
LLM context windows are limited and expensive. More importantly, a vector search works best when
each chunk represents one coherent idea. If a chunk is an entire chapter, the vector for that chunk
is an average of hundreds of ideas and matches nothing precisely.

**Why `RecursiveCharacterTextSplitter`?**
It is "recursive" because it tries separators in order. First it tries to split on `"\n\n"` (paragraph
breaks). If a piece is still too large, it tries `"\n"` (line breaks). Then `". "` (sentences).
Then `" "` (words). Then `""` (characters as last resort).

This produces natural, semantically coherent chunks because it respects the document's own structure.

**Example — what chunking looks like:**

Input page (simplified):
```
The Transformer uses an encoder-decoder structure.

The encoder maps the input sequence to a continuous representation.
The decoder then generates the output sequence one token at a time.

Each layer uses multi-head attention and a feed-forward network.
```

After chunking with size=500, overlap=50, you might get:
```
Chunk 0: "The Transformer uses an encoder-decoder structure.\n\nThe encoder maps
          the input sequence to a continuous representation. The decoder then
          generates the output sequence one token at a time."

Chunk 1: "The decoder then generates the output sequence one token at a time.
          \n\nEach layer uses multi-head attention and a feed-forward network."
```

Notice Chunk 1 repeats the last sentence of Chunk 0 — that is the overlap. It ensures that if a
sentence spans the boundary between two chunks, it is fully present in at least one of them.

**Alternatives considered:**
- **Fixed-size character split** — simpler but cuts mid-sentence constantly.
- **Sentence splitter (NLTK/spaCy)** — very clean boundaries but requires NLP models and is slower.
- **Semantic chunking** — groups sentences by meaning using embeddings. State of the art but slow
  and complex. Overkill for a baseline system.
- **Page-level (no chunking)** — pages are too big and too generic for precise retrieval.

---

### 4.3 Embedding

```python
from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

embedder = get_embedder()
texts = [c.page_content for c in chunks]
embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
```

Embedding converts text into a list of numbers (a vector) that captures its meaning.

**What does a vector look like?**

The model `all-MiniLM-L6-v2` produces 384-dimensional vectors:

```python
embedder.encode(["attention is all you need"])
# → array([-0.0234,  0.1823, -0.0941,  0.2103, ...])  ← 384 numbers
```

Two texts that mean the same thing will have vectors that point in similar directions. Two texts
about different topics will have vectors that point in very different directions. This is how we do
"semantic search" — we search by meaning, not keywords.

**Why `all-MiniLM-L6-v2`?**
- Runs entirely locally — no API key, no network call, no cost per embedding.
- Small (80 MB) and fast.
- Good quality for general English text.
- Free open-source model from the `sentence-transformers` library.

**Alternatives considered:**
- **OpenAI `text-embedding-ada-002`** — slightly better quality but costs money per token, requires
  internet, and ties you to OpenAI. Privacy concern for sensitive documents.
- **Cohere `embed-english-v3.0`** — similar trade-off: paid, requires internet.
- **`all-mpnet-base-v2`** — larger (420 MB), slightly better quality, but 5x slower than MiniLM.
  Not worth it for a baseline.
- **BERT base** — older, not optimized for sentence similarity tasks. MiniLM was specifically
  trained for this purpose.

**The singleton pattern explained:**

```python
_embedder: SentenceTransformer | None = None   # module-level variable, starts as None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:          # first call: load the model (takes ~1 second)
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder               # subsequent calls: return the already-loaded model
```

Loading the model takes about 1 second and uses ~300 MB of memory. Without the singleton, every
PDF ingestion and every query would reload the model from disk. With the singleton, it loads once
when the server starts and stays in memory. Both `ingestion.py` and `retrieval.py` call `get_embedder()`
from `ingestion.py` — the same instance is returned both times, guaranteeing they always use the
same model.

**This is critical:** if ingestion used model A and retrieval used model B, the vectors would be
in completely different "spaces" and similarity scores would be meaningless garbage.

---

### 4.4 Vector Storage (ChromaDB)

```python
import chromadb

_chroma_client: chromadb.PersistentClient | None = None

def get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
```

ChromaDB is the vector database — it stores the text, the vectors, and the metadata (source, page
number, chunk index), and it can search them by vector similarity.

**What gets stored for each chunk:**

```python
collection.upsert(
    ids=["a3f4..."],                          # unique ID per chunk
    embeddings=[[-0.023, 0.182, ...]],        # 384-dimensional vector
    documents=["The encoder maps..."],         # original text
    metadatas=[{"source": "paper.pdf",        # citation info
                "page": "1",
                "chunk_index": "3"}]
)
```

**Why `upsert` instead of `insert`?**
`upsert` = update if exists, insert if not. If you upload the same PDF twice, the chunks get
overwritten in place (same ID → same slot). Without upsert, you'd accumulate duplicate chunks
and get duplicate results in every search.

**Why `"hnsw:space": "cosine"`?**
This tells ChromaDB to use cosine similarity as the distance metric. HNSW (Hierarchical Navigable
Small World) is the indexing algorithm. Cosine similarity measures the angle between two vectors —
it works well for text because it ignores length (a long paragraph and a short sentence about the
same topic will still score highly).

**Why ChromaDB?**

| Database | Pros | Cons |
|---|---|---|
| **ChromaDB** ✓ | Runs locally, zero config, Python-native | Not for billions of vectors |
| **FAISS** | Extremely fast, battle-tested | No metadata filtering, no persistence by default, harder API |
| **Pinecone** | Managed, scalable, fast | Paid, requires internet, external dependency |
| **Weaviate** | Full-featured, GraphQL API | Heavy (needs Docker), complex setup |
| **Qdrant** | Fast, good API | Requires separate server process |
| **pgvector** | If you already use Postgres | Requires Postgres setup |

ChromaDB wins for a local development system because it is one `pip install` and zero config.
The database is just a folder on disk.

---

### 4.5 Deterministic Chunk IDs

```python
import hashlib

def _doc_id(text: str, source: str, chunk_index: int) -> str:
    digest = hashlib.md5(f"{source}:{chunk_index}:{text[:80]}".encode()).hexdigest()
    return digest
```

**Example:**
```python
_doc_id("The encoder maps the input...", "paper.pdf", 3)
# → "a3f4b2c1d9e8f7a6b5c4d3e2f1a0b9c8"  (always the same for the same inputs)
```

The ID is built from three things:
1. `source` — the filename
2. `chunk_index` — position in the document
3. `text[:80]` — first 80 characters of the text (catches content changes)

**Why MD5?** We are not using it for security (MD5 is broken for that). We use it as a fast
"fingerprinting" function that converts an arbitrary string into a fixed-length hex string suitable
as a database key. It is deterministic — same inputs always produce the same output.

**Why not just use `source + "_" + chunk_index`?**
That would work for stable documents, but if the same source file produced different text (e.g.,
due to a PDF update), the ID would collide and silently overwrite with wrong data. The text[:80]
adds content-awareness.

---

## 5. Retrieval Pipeline — retrieval.py

When a user asks a question, this sequence runs:

```
question  →  embed  →  vector search  →  top 4 chunks
          →  format as context string
          →  fill prompt template
          →  send to Claude API
          →  return answer + sources
```

### 5.1 Query Embedding

```python
embedder = get_embedder()
query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]
```

The question gets embedded with the **exact same model** used during ingestion. This is mandatory —
the question vector and the chunk vectors must live in the same 384-dimensional space for similarity
to be meaningful.

**Example:**
```python
query = "How many attention heads does the Transformer use?"
query_embedding = embedder.encode([query]).tolist()[0]
# → [-0.041, 0.217, -0.088, ...]  ← 384 numbers
```

---

### 5.2 Cosine Search

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(TOP_K_RESULTS, collection.count()),
    include=["documents", "metadatas", "distances"],
)
```

ChromaDB compares the query vector against every stored chunk vector using cosine distance, ranks
them, and returns the top K.

**What ChromaDB returns:**
```python
{
  "documents": [["The base model uses h=8 parallel...", "Multi-head attention allows...", ...]],
  "metadatas": [[{"source": "paper.pdf", "page": "4", ...}, ...]],
  "distances": [[0.12, 0.18, 0.31, 0.45]]   # cosine DISTANCES (lower = more similar)
}
```

**Distance to similarity conversion:**
```python
similarity = 1 - distance
# distance 0.12 → similarity 0.88  (very relevant)
# distance 0.45 → similarity 0.55  (somewhat relevant)
```

ChromaDB returns **distances**, not similarities. We subtract from 1 to make high scores mean
"more relevant", which is more intuitive to display.

---

### 5.3 Context Formatting

```python
def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

**What the formatted context looks like:**
```
[paper.pdf, page 4]
The base model uses h=8 parallel attention heads. For each head, we use
dk = dv = dmodel/h = 64 dimensions.

---

[paper.pdf, page 4]
Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions.

---

[paper.pdf, page 5]
...
```

This string becomes the `{context}` variable in the prompt. The `[source, page N]` labels teach
the LLM where each piece of information came from, so it can produce citations.

---

### 5.4 Prompt Template

```python
from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful assistant answering questions based solely on the provided context.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Always be concise and cite the source document and page when referencing specific facts."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer using only the context above. Include citations like [source.pdf, page N] inline."""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)
```

**Why a system prompt + human prompt split?**
Claude's API uses a "messages" format with roles. The system message sets persistent instructions
(who the assistant is, what rules it follows). The human message contains the per-request data
(the context and question). Keeping them separate is cleaner and follows the API's intended design.

**Why "answer solely from the context"?**
Without this instruction, Claude might supplement the retrieved chunks with its own training
knowledge. That would produce answers that appear sourced but aren't — a hallucination risk.
Grounding the LLM to the context makes citations meaningful and keeps the system honest.

**Why `ChatPromptTemplate` from LangChain instead of f-strings?**
LangChain's template handles escaping, validates that `{context}` and `{question}` are filled,
and produces the correct message format for the Claude API. With raw f-strings you'd have to
build the messages list yourself.

---

### 5.5 The LLM Call

```python
def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        anthropic_api_key=ANTHROPIC_API_KEY,
        timeout=60,
    )

chain = _prompt | _get_llm()
response = chain.invoke({"context": context, "question": question})
answer_text = response.content
```

**The `|` (pipe) operator — LangChain chains:**

```python
chain = _prompt | _get_llm()
```

This creates a pipeline. `chain.invoke({"context": ..., "question": ...})` does:
1. Fill `_prompt` with the variables → produces a list of messages
2. Pass those messages to `_get_llm()` → sends to Claude API
3. Return the response object

This is LangChain's "LCEL" (LangChain Expression Language). It makes pipelines composable and
readable.

**Why is `_get_llm()` called per request (not a singleton)?**
The LLM object holds no important state — it is just a client configured with a model name and
API key. Creating it per-request is cheap (no network call happens at creation time, only at
`.invoke()`). This means you can change `LLM_MODEL` in `config.py` and the next request will use
the new model without a server restart.

**Why Claude over GPT-4 / Gemini?**
This is an Anthropic-built project, so Claude is the natural choice. Claude Sonnet offers an
excellent quality-to-cost ratio and has a large 200K token context window — useful if you wanted
to increase `TOP_K_RESULTS` significantly.

---

## 6. API Server — main.py

```python
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="RAG QA System", version="1.0.0")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ...

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    ...
```

**Why FastAPI?**

| Framework | Pros | Cons |
|---|---|---|
| **FastAPI** ✓ | Auto docs at `/docs`, async support, Pydantic validation, fast | |
| **Flask** | Simple, widely known | No async, no automatic validation |
| **Django** | Full-featured | Way too heavy for a 2-endpoint API |
| **aiohttp** | Pure async | Lower-level, more boilerplate |

FastAPI automatically generates an interactive API explorer at `http://localhost:8000/docs`. You
can test the endpoints directly in your browser without writing any code.

**The `/ingest` endpoint — file size check:**

```python
chunk = await file.read(MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
if len(chunk) > MAX_FILE_SIZE_MB * 1024 * 1024:
    raise HTTPException(status_code=413, ...)
```

This reads one byte more than the limit. If the read returned more data than the limit, the file
is too big. This avoids loading a 10 GB file into memory before discovering it's too large.

**Why `async def ingest` but `def query` (not async)?**
File upload is I/O-bound — it spends most of its time waiting for bytes to arrive from the network.
`async def` lets FastAPI handle other requests while waiting. The query endpoint, however, calls
synchronous libraries (ChromaDB, SentenceTransformers) that don't support `async`, so it is
declared as a regular `def`. FastAPI runs sync endpoints in a thread pool automatically.

**CORS middleware:**
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
This allows the frontend (running on port 8501) to make HTTP requests to the backend (port 8000).
Browsers block cross-origin requests by default; this header tells them it's allowed.

---

## 7. Frontend — frontend/app.py

```python
import streamlit as st

st.set_page_config(page_title="RAG QA System", layout="wide")
```

**Why Streamlit?**

| Option | Pros | Cons |
|---|---|---|
| **Streamlit** ✓ | Pure Python, minimal code, built-in state, rapid prototyping | Not for complex UI |
| **React** | Full control, production-grade | Requires JavaScript, build step, much more code |
| **Gradio** | Also Python, easy | Less flexible layout |
| **Flask + HTML** | Simple, flexible | Need to write HTML/CSS/JS |

Streamlit lets you build an interactive web UI in pure Python with almost no boilerplate. The
entire frontend is ~60 lines.

**Session state — how chat history works:**

```python
if "history" not in st.session_state:
    st.session_state.history = []
```

Streamlit reruns your entire script from top to bottom on every user interaction (button click,
text input, etc.). Without `st.session_state`, all variables would reset to empty on every
interaction. `session_state` is a dictionary that persists across reruns within a browser session.

```python
# After a successful query:
st.session_state.history.append({
    "question": question,
    "answer": data["answer"],
    "sources": data["sources"]
})
st.rerun()  # triggers a rerun, which re-draws the chat with the new message
```

**Two-column layout:**

```python
chat_col, source_col = st.columns([3, 2])
```

The `[3, 2]` ratio means the chat column takes 3/5 of the width and the source panel takes 2/5.
This mirrors how citation-heavy research tools are laid out — main content on the left, references
on the right.

**The source panel:**

```python
with st.expander(f"[{i}] {src['source']} — page {src['page']}  (score: {score:.3f})"):
    st.caption(f"Chunk #{src.get('chunk_index', '?')}")
    st.write(src.get("excerpt", ""))
```

Each source chunk is shown as a collapsible expander. The similarity score is displayed so the
user can see how confident the retrieval was. A score of 0.9+ means very relevant. Below 0.6
suggests the document may not contain the answer.

---

## 8. Evaluation System — evals/

The eval system answers: "How good is this RAG system actually?"

### Keyword matching (fast, free)

```python
def keyword_hit(answer: str, keywords: list[str]) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)
```

For each question, we have a list of expected keywords:

```python
{
    "id": "q03",
    "question": "How many attention heads are used in the base model?",
    "expected_keywords": ["8"],
}
```

If the answer contains "8", the test passes. This is simple but catches obvious failures
(hallucination, wrong document, empty retrieval).

**Example eval run:**
```
  [q03] PASS    ← answer contains "8"
  [q07] FAIL    ← answer doesn't mention "28" or "BLEU"  (retrieval probably missed that page)
  [q15] PASS    ← answer contains "P100" (which also contains "p100" lowercased)
```

### Hallucination detection

```python
hallucination_flag = not kw_pass and retrieval_ok
```

If retrieval found chunks (so there was relevant content) but the answer doesn't contain the
expected keywords, it suggests the LLM may have made up an answer rather than using the context.
This is a heuristic, not proof.

### Ragas metrics (slow, requires API)

```python
from ragas.metrics import answer_faithfulness, context_recall, context_precision
result = evaluate(dataset, metrics=[context_precision, context_recall, answer_faithfulness])
```

Ragas uses an LLM to judge quality on three dimensions:

| Metric | What it measures | Example |
|---|---|---|
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | If you asked about attention heads but retrieved chunks about optimizer settings, precision is low. |
| **Context Recall** | Did retrieval find all the chunks needed to answer correctly? | If the answer requires 3 facts but only 1 was retrieved, recall is low. |
| **Answer Faithfulness** | Is the answer grounded in the retrieved context, or did the LLM add things not in the context? | If the context says "8 heads" but the answer says "16 heads", faithfulness is 0. |

These metrics require calling an LLM to judge each response, which is why they cost money and
take time. That's why `--skip-ragas` exists for quick checks.

---

## 9. Docker & Deployment

### Two containers

```yaml
# docker-compose.yml
services:
  backend:
    build: { dockerfile: ../docker/Dockerfile.backend }
    ports: [ "8000:8000" ]
    volumes:
      - chroma_data:/data/chroma_db    # persists the vector database
      - upload_data:/app/uploads        # persists uploaded PDFs

  frontend:
    build: { dockerfile: ../docker/Dockerfile.frontend }
    ports: [ "8501:8501" ]
    depends_on:
      backend:
        condition: service_healthy     # waits for backend to be ready
```

**Why two separate containers?**
Separation of concerns. If you want to scale (run 3 backend instances behind a load balancer), you
can do that without touching the frontend. You can also update the frontend without restarting the
backend (and losing cached embeddings).

**The health check:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s
```

The backend might take 20-30 seconds to start (it loads the embedding model on first startup). The
health check polls `/health` every 30 seconds. The frontend container won't start until the backend
passes 1 health check. Without this, the frontend would start, immediately try to reach the backend,
fail, and show an error.

**Named volumes — persistence:**
```yaml
volumes:
  chroma_data:    # docker manages this directory
  upload_data:
```

Docker named volumes persist data between container restarts. If you stop and restart the containers,
your ingested documents are still there. If you used a bind mount (`./chroma_db:/data/chroma_db`)
instead, the data would be in your working directory on the host.

**Pre-downloading the embedding model in the Dockerfile:**
```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

This runs during `docker build`, not at container startup. The model is baked into the image.
Without this, the first startup would need to download ~80 MB from the internet, adding 30-60
seconds to the first request and failing entirely if there's no internet.

---

## 10. End-to-End Trace: One Question, Every Step

Let's trace the question: **"How many attention heads does the Transformer use?"**

Assuming "Attention Is All You Need" (Vaswani et al.) has already been ingested.

---

**Step 1 — User types the question in the Streamlit UI**

`frontend/app.py` calls:
```python
resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "How many attention heads does the Transformer use?"},
    timeout=120,
)
```

---

**Step 2 — FastAPI receives the request**

`backend/main.py`:
```python
@app.post("/query")
def query(request: QueryRequest):
    result = answer_query(request.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
```

It validates the request (Pydantic ensures `question` is a non-empty string) and calls
`answer_query`.

---

**Step 3 — Embed the question**

`retrieval.py → retrieve_chunks()`:
```python
query_embedding = embedder.encode(
    ["How many attention heads does the Transformer use?"]
).tolist()[0]
# → [-0.041, 0.217, -0.088, 0.156, ...]  ← 384 numbers
```

---

**Step 4 — Search ChromaDB**

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4,
    include=["documents", "metadatas", "distances"],
)
```

ChromaDB computes cosine distance from the query vector to every stored chunk vector (there might
be ~200 chunks from a 15-page paper). It returns the 4 closest:

```python
# Simplified result:
{
  "documents": [[
    "The base model uses h=8 parallel attention heads...",        # distance 0.08
    "Multi-head attention runs h attention functions in parallel...", # distance 0.14
    "We use h=8 heads and dk=dv=64 dimensions for each head.",   # distance 0.17
    "Attention allows the model to focus on different positions..."  # distance 0.41
  ]],
  "distances": [[0.08, 0.14, 0.17, 0.41]]
}
```

---

**Step 5 — Format the context**

```python
context = format_context(docs)
```

Produces:
```
[attention_paper.pdf, page 4]
The base model uses h=8 parallel attention heads. For each of these
we use dk = dv = dmodel/h = 64.

---

[attention_paper.pdf, page 4]
Multi-head attention runs h attention functions in parallel on queries,
keys, and values of lower dimensions.

---

[attention_paper.pdf, page 5]
We use h=8 heads and dk=dv=64 dimensions for each head.

---

[attention_paper.pdf, page 3]
Attention allows the model to focus on different positions across the sequence.
```

---

**Step 6 — Fill the prompt and call Claude**

```python
chain = _prompt | _get_llm()
response = chain.invoke({
    "context": context,
    "question": "How many attention heads does the Transformer use?"
})
```

The actual request sent to the Claude API looks like:
```
System: You are a helpful assistant answering questions based solely on the provided context.
        If the answer is not contained in the context, say "I don't have enough information..."
        Always be concise and cite the source document and page.

Human: Context:
[attention_paper.pdf, page 4]
The base model uses h=8 parallel attention heads...

Question: How many attention heads does the Transformer use?

Answer using only the context above. Include citations like [source.pdf, page N] inline.
```

---

**Step 7 — Claude responds**

```
The Transformer base model uses **8 parallel attention heads** [attention_paper.pdf, page 4].
Each head operates on dk = dv = dmodel/h = 64 dimensions.
```

`response.content` is a plain string — LangChain unwraps the API response object for you.

---

**Step 8 — Assemble and return the response**

```python
sources = [
    {
        "source": "attention_paper.pdf",
        "page": "4",
        "chunk_index": "23",
        "similarity_score": 0.92,
        "excerpt": "The base model uses h=8 parallel attention heads..."
    },
    ...  # 3 more sources
]
return {"answer": answer_text, "sources": sources}
```

FastAPI serialises this to JSON and sends it back to Streamlit.

---

**Step 9 — Streamlit displays the result**

The user now sees:
- **Chat column:** Their question and Claude's answer with inline citations.
- **Source column:** 4 expandable cards, each showing filename, page, similarity score, and the first 200 characters of the chunk.

---

## 11. Glossary

| Term | Plain English |
|---|---|
| **RAG** | Retrieval-Augmented Generation. Find relevant text first, then ask the LLM to answer using only that text. |
| **Embedding** | A list of numbers representing the meaning of text. Similar texts have similar numbers. |
| **Vector** | Another word for an embedding. A point in high-dimensional space. |
| **Vector database** | A database optimised to store and search vectors by similarity. ChromaDB is one example. |
| **Cosine similarity** | Measures how similar two vectors are by the angle between them. 1.0 = identical direction, 0.0 = unrelated. |
| **Chunk** | A small piece of a document, typically 3-5 sentences. The unit of storage and retrieval. |
| **Chunk overlap** | Shared characters between adjacent chunks, preventing sentences from being cut in half at boundaries. |
| **Singleton** | A programming pattern where a class is instantiated once and every caller gets the same instance. |
| **Upsert** | Insert if new, update if exists. Used so re-ingesting the same PDF does not create duplicates. |
| **HNSW** | Hierarchical Navigable Small World. The indexing algorithm ChromaDB uses for fast nearest-neighbour search. |
| **LangChain** | A Python library providing building blocks for LLM apps: prompt templates, chains, document loaders, text splitters. |
| **LCEL** | LangChain Expression Language. The `|` pipe syntax for chaining steps together. |
| **FastAPI** | A Python web framework for building HTTP APIs, with automatic validation and interactive docs. |
| **Streamlit** | A Python library for building interactive web UIs with no HTML or JavaScript required. |
| **Temperature** | An LLM setting controlling randomness. 0.0 = deterministic and factual. 1.0 = creative and varied. |
| **Hallucination** | When an LLM confidently states something false or not supported by the provided context. |
| **Faithfulness** | A Ragas metric measuring whether the LLM's answer is grounded in the retrieved context. |
| **Context Precision** | A Ragas metric measuring whether retrieved chunks are actually relevant to the question. |
| **Context Recall** | A Ragas metric measuring whether all necessary information was retrieved. |
| **CORS** | Cross-Origin Resource Sharing. Browser security that controls which domains can call your API. |
| **Named volume** | A Docker-managed persistent storage location that survives container restarts. |
| **Health check** | A periodic test Docker runs to confirm a container is ready to serve requests. |
| **Deterministic ID** | An ID computed from content so that the same input always produces the same ID. |
