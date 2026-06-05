import logging
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RESULTS
from ingestion import get_collection, get_embedder

logger = logging.getLogger(__name__)

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


def _get_llm() -> ChatAnthropic:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    return ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=ANTHROPIC_API_KEY,
        timeout=60,
    )


def retrieve_chunks(query: str) -> list[Document]:
    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_RESULTS, count),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append(
            Document(
                page_content=text,
                metadata={**meta, "similarity_score": round(1 - dist, 4)},
            )
        )
    return docs


def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_query(question: str) -> dict[str, Any]:
    """Retrieve relevant chunks and generate an LLM answer with citations."""
    try:
        docs = retrieve_chunks(question)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return {"answer": f"Retrieval error: {exc}", "sources": []}

    if not docs:
        return {
            "answer": "No relevant documents found. Please ingest a PDF first.",
            "sources": [],
        }

    context = format_context(docs)
    chain = _prompt | _get_llm()

    try:
        response = chain.invoke({"context": context, "question": question})
        answer_text = response.content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {"answer": f"LLM error: {exc}", "sources": [doc.metadata for doc in docs]}

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?"),
            "chunk_index": d.metadata.get("chunk_index", "?"),
            "similarity_score": d.metadata.get("similarity_score", 0.0),
            "excerpt": d.page_content[:200],
        }
        for d in docs
    ]
    return {"answer": answer_text, "sources": sources}
