from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import main

    return TestClient(main.app)


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── POST /ingest ───────────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf_file(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_ingest_rejects_missing_file(client):
    r = client.post("/ingest")
    assert r.status_code == 422


def test_ingest_rejects_oversized_file(client):
    import main

    with patch.object(main, "MAX_FILE_SIZE_MB", 0):
        r = client.post(
            "/ingest",
            files={"file": ("big.pdf", b"x" * 10, "application/pdf")},
        )
    assert r.status_code == 413


def test_ingest_success_returns_filename_and_chunk_count(client, tmp_path):
    import main

    mock_result = {"filename": "paper.pdf", "chunks_ingested": 42}
    with (
        patch.object(main, "UPLOAD_DIR", tmp_path),
        patch("main.ingest_pdf", return_value=mock_result),
    ):
        r = client.post(
            "/ingest",
            files={"file": ("paper.pdf", b"%PDF-1.4 content", "application/pdf")},
        )
    assert r.status_code == 200
    assert r.json() == mock_result


def test_ingest_propagates_ingest_pdf_error(client, tmp_path):
    import main

    with (
        patch.object(main, "UPLOAD_DIR", tmp_path),
        patch("main.ingest_pdf", side_effect=ValueError("PDF appears to be empty")),
    ):
        r = client.post(
            "/ingest",
            files={"file": ("bad.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert r.status_code == 500
    assert "PDF appears to be empty" in r.json()["detail"]


# ── POST /query ────────────────────────────────────────────────────────────────

def test_query_rejects_empty_question(client):
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 400


def test_query_rejects_whitespace_only_question(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400


def test_query_rejects_missing_body(client):
    r = client.post("/query")
    assert r.status_code == 422


def test_query_returns_no_documents_message_when_nothing_ingested(client):
    no_docs = {"answer": "No relevant documents found. Please ingest a PDF first.", "sources": []}
    with patch("main.answer_query", return_value=no_docs):
        r = client.post("/query", json={"question": "What is attention?"})
    assert r.status_code == 200
    assert "No relevant documents" in r.json()["answer"]
    assert r.json()["sources"] == []


def test_query_success_returns_answer_and_sources(client):
    mock_result = {
        "answer": "The transformer uses multi-head self-attention.",
        "sources": [
            {
                "source": "paper.pdf",
                "page": "3",
                "chunk_index": "0",
                "similarity_score": 0.95,
                "excerpt": "multi-head attention mechanism",
            }
        ],
    }
    with patch("main.answer_query", return_value=mock_result):
        r = client.post("/query", json={"question": "What does the transformer use?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "The transformer uses multi-head self-attention."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["similarity_score"] == 0.95
