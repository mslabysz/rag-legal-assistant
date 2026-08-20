import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from rag_legal_assistant.api.server import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_qdrant(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "ok"


def test_documents_endpoint_lists_only_pdfs(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert all(name.endswith(".pdf") for name in response.json()["documents"])


def test_upload_rejects_non_pdf_extension(client):
    response = client.post("/upload", files={"file": ("notatka.txt", b"nie pdf", "text/plain")})

    assert response.status_code == 400


def test_upload_rejects_corrupted_pdf(client):
    response = client.post(
        "/upload", files={"file": ("uszkodzony.pdf", b"to nie jest pdf", "application/pdf")}
    )

    assert response.status_code == 400


def test_upload_rejects_path_traversal(client):
    response = client.post(
        "/upload", files={"file": ("../../etc/zly.pdf", b"to nie jest pdf", "application/pdf")}
    )

    assert response.status_code == 400


def test_chat_rejects_blank_query(client):
    response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 422