import retrieval as retrieval_module
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document


def _make_doc(content="some text", source="doc.pdf", page="1", chunk_index="0", similarity=0.9):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "similarity_score": similarity,
        },
    )


# ── format_context ─────────────────────────────────────────────────────────────

def test_format_context_empty_list():
    from retrieval import format_context

    assert format_context([]) == ""


def test_format_context_single_doc_contains_header_and_body():
    from retrieval import format_context

    doc = Document(page_content="This is test content", metadata={"source": "paper.pdf", "page": "3"})
    result = format_context([doc])
    assert "[paper.pdf, page 3]" in result
    assert "This is test content" in result


def test_format_context_multiple_docs_are_separated():
    from retrieval import format_context

    docs = [
        Document(page_content="text one", metadata={"source": "a.pdf", "page": "1"}),
        Document(page_content="text two", metadata={"source": "b.pdf", "page": "2"}),
    ]
    result = format_context(docs)
    assert "---" in result
    assert "text one" in result
    assert "text two" in result


def test_format_context_missing_metadata_uses_defaults():
    from retrieval import format_context

    doc = Document(page_content="content", metadata={})
    result = format_context([doc])
    assert "unknown" in result
    assert "?" in result


# ── retrieve_chunks ────────────────────────────────────────────────────────────

def test_retrieve_chunks_returns_empty_when_collection_is_empty():
    from retrieval import retrieve_chunks

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(**{"tolist.return_value": [[0.1] * 384]})
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with (
        patch("retrieval.get_embedder", return_value=mock_embedder),
        patch("retrieval.get_collection", return_value=mock_collection),
    ):
        result = retrieve_chunks("any question")

    assert result == []
    mock_collection.query.assert_not_called()


def test_retrieve_chunks_returns_documents_with_similarity_score():
    from retrieval import retrieve_chunks

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(**{"tolist.return_value": [[0.1] * 384]})
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["attention is the key mechanism"]],
        "metadatas": [[{"source": "paper.pdf", "page": "3", "chunk_index": "0"}]],
        "distances": [[0.15]],
    }

    with (
        patch("retrieval.get_embedder", return_value=mock_embedder),
        patch("retrieval.get_collection", return_value=mock_collection),
    ):
        docs = retrieve_chunks("what is attention?")

    assert len(docs) == 1
    assert docs[0].page_content == "attention is the key mechanism"
    assert docs[0].metadata["similarity_score"] == round(1 - 0.15, 4)


def test_retrieve_chunks_similarity_is_one_minus_distance():
    from retrieval import retrieve_chunks

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(**{"tolist.return_value": [[0.1] * 384]})
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["text"]],
        "metadatas": [[{"source": "x.pdf", "page": "1", "chunk_index": "0"}]],
        "distances": [[0.3]],
    }

    with (
        patch("retrieval.get_embedder", return_value=mock_embedder),
        patch("retrieval.get_collection", return_value=mock_collection),
    ):
        docs = retrieve_chunks("question")

    assert docs[0].metadata["similarity_score"] == round(1 - 0.3, 4)


# ── answer_query ───────────────────────────────────────────────────────────────

def test_answer_query_returns_no_documents_message_when_collection_empty():
    from retrieval import answer_query

    with patch("retrieval.retrieve_chunks", return_value=[]):
        result = answer_query("what is attention?")

    assert "No relevant documents" in result["answer"]
    assert result["sources"] == []


def test_answer_query_handles_retrieval_error_gracefully():
    from retrieval import answer_query

    with patch("retrieval.retrieve_chunks", side_effect=RuntimeError("db offline")):
        result = answer_query("test question")

    assert "Retrieval error" in result["answer"]
    assert result["sources"] == []


def test_answer_query_handles_llm_error_gracefully():
    from retrieval import answer_query

    mock_doc = _make_doc()
    with (
        patch("retrieval.retrieve_chunks", return_value=[mock_doc]),
        patch("retrieval._get_llm", side_effect=EnvironmentError("no API key")),
    ):
        result = answer_query("test question")

    assert "LLM error" in result["answer"]
    assert len(result["sources"]) == 1


def test_answer_query_success_returns_answer_and_sources():
    from retrieval import answer_query

    mock_doc = _make_doc(
        content="The transformer relies on multi-head self-attention.",
        source="attention.pdf",
        page="3",
        similarity=0.95,
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Transformers use multi-head attention.")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch("retrieval.retrieve_chunks", return_value=[mock_doc]),
        patch.object(retrieval_module, "_prompt", mock_prompt),
        patch("retrieval._get_llm"),
    ):
        result = answer_query("What does the transformer use?")

    assert result["answer"] == "Transformers use multi-head attention."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "attention.pdf"
    assert result["sources"][0]["similarity_score"] == 0.95


def test_answer_query_source_excerpt_is_truncated():
    from retrieval import answer_query

    long_text = "x" * 500
    mock_doc = _make_doc(content=long_text)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="The answer.")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch("retrieval.retrieve_chunks", return_value=[mock_doc]),
        patch.object(retrieval_module, "_prompt", mock_prompt),
        patch("retrieval._get_llm"),
    ):
        result = answer_query("question")

    assert len(result["sources"][0]["excerpt"]) == 200
