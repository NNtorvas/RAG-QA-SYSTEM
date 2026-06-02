import hashlib
import logging
from pathlib import Path

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _doc_id(text: str, source: str, chunk_index: int) -> str:
    digest = hashlib.md5(f"{source}:{chunk_index}:{text[:80]}".encode()).hexdigest()
    return digest


def ingest_pdf(pdf_path: Path) -> dict:
    """Load a PDF, chunk it, embed chunks, and upsert into ChromaDB."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    raw_docs = loader.load()
    if not raw_docs:
        raise ValueError("PDF appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        raise ValueError("No text chunks produced from PDF.")

    embedder = get_embedder()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection = get_collection()
    ids = [_doc_id(t, pdf_path.name, i) for i, t in enumerate(texts)]
    metadatas = [
        {
            "source": pdf_path.name,
            "page": str(c.metadata.get("page", "?")),
            "chunk_index": str(i),
        }
        for i, c in enumerate(chunks)
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Ingested %d chunks from %s", len(chunks), pdf_path.name)
    return {"filename": pdf_path.name, "chunks_ingested": len(chunks)}
