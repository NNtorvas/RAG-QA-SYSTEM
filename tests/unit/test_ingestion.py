import hashlib
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document


def test_doc_id_is_deterministic():
    from ingestion import _doc_id

    assert _doc_id("hello", "doc.pdf", 0) == _doc_id("hello", "doc.pdf", 0)


def test_doc_id_matches_md5():
    from ingestion import _doc_id

    text, source, idx = "hello world", "doc.pdf", 0
    expected = hashlib.md5(f"{source}:{idx}:{text[:80]}".encode()).hexdigest()
    assert _doc_id(text, source, idx) == expected


def test_doc_id_differs_by_source():
    from ingestion import _doc_id

    assert _doc_id("same text", "a.pdf", 0) != _doc_id("same text", "b.pdf", 0)


def test_doc_id_differs_by_chunk_index():
    from ingestion import _doc_id

    assert _doc_id("same text", "doc.pdf", 0) != _doc_id("same text", "doc.pdf", 1)


def test_get_embedder_is_singleton(monkeypatch):
    import ingestion

    mock_model = MagicMock()
    monkeypatch.setattr(ingestion, "_embedder", None)
    with patch("ingestion.SentenceTransformer", return_value=mock_model) as mock_st:
        e1 = ingestion.get_embedder()
        e2 = ingestion.get_embedder()
        assert e1 is e2
        mock_st.assert_called_once()


def test_get_collection_uses_cosine_space(monkeypatch):
    import ingestion

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    monkeypatch.setattr(ingestion, "_chroma_client", None)
    with patch("chromadb.PersistentClient", return_value=mock_client):
        col = ingestion.get_collection()

    assert col is mock_collection
    mock_client.get_or_create_collection.assert_called_once_with(
        name=ingestion.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def test_ingest_pdf_raises_when_file_missing(tmp_path):
    from ingestion import ingest_pdf

    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "nonexistent.pdf")


def test_ingest_pdf_raises_on_empty_pdf(tmp_path):
    from ingestion import ingest_pdf

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()

    with patch("ingestion.PyPDFLoader") as mock_loader_cls:
        mock_loader_cls.return_value.load.return_value = []
        with pytest.raises(ValueError, match="empty or unreadable"):
            ingest_pdf(pdf_path)


def test_ingest_pdf_raises_when_no_chunks_produced(tmp_path):
    from ingestion import ingest_pdf

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf content")
    fake_docs = [Document(page_content="some content", metadata={"page": 0})]

    with (
        patch("ingestion.PyPDFLoader") as mock_loader_cls,
        patch("ingestion.RecursiveCharacterTextSplitter") as mock_splitter_cls,
    ):
        mock_loader_cls.return_value.load.return_value = fake_docs
        mock_splitter_cls.return_value.split_documents.return_value = []
        with pytest.raises(ValueError, match="No text chunks"):
            ingest_pdf(pdf_path)


def test_ingest_pdf_success(tmp_path):
    from ingestion import ingest_pdf

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    fake_docs = [Document(page_content="Attention is all you need.", metadata={"page": 0})]
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(**{"tolist.return_value": [[0.1] * 384]})
    mock_collection = MagicMock()

    with (
        patch("ingestion.PyPDFLoader") as mock_loader_cls,
        patch("ingestion.get_embedder", return_value=mock_embedder),
        patch("ingestion.get_collection", return_value=mock_collection),
    ):
        mock_loader_cls.return_value.load.return_value = fake_docs
        result = ingest_pdf(pdf_path)

    assert result["filename"] == "paper.pdf"
    assert result["chunks_ingested"] >= 1
    mock_collection.upsert.assert_called_once()


def test_ingest_pdf_upsert_receives_correct_metadata(tmp_path):
    from ingestion import ingest_pdf

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    fake_docs = [Document(page_content="Some chunk text here.", metadata={"page": 2})]
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(**{"tolist.return_value": [[0.2] * 384]})
    mock_collection = MagicMock()

    with (
        patch("ingestion.PyPDFLoader") as mock_loader_cls,
        patch("ingestion.get_embedder", return_value=mock_embedder),
        patch("ingestion.get_collection", return_value=mock_collection),
    ):
        mock_loader_cls.return_value.load.return_value = fake_docs
        ingest_pdf(pdf_path)

    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert call_kwargs["metadatas"][0]["source"] == "paper.pdf"
    assert call_kwargs["metadatas"][0]["page"] == "2"
    assert call_kwargs["metadatas"][0]["chunk_index"] == "0"
