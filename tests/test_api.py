import os
import io
import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

SAMPLE_DOCX = os.path.join(os.path.dirname(__file__), "split_run_sample.docx")


def test_health_endpoint():
    """Hinglish: Verify that /health endpoint is working and returns standard status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_frontend():
    """Hinglish: Verify that / route serves the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>PII Redactor" in response.text


def test_redact_endpoint_with_valid_docx():
    """Hinglish: Verify /redact endpoint with a valid small docx card."""
    if not os.path.exists(SAMPLE_DOCX):
        pytest.skip("split_run_sample.docx not found in tests folder.")
        
    with open(SAMPLE_DOCX, "rb") as f:
        response = client.post(
            "/redact",
            files={"file": ("split_run_sample.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
    assert response.status_code == 200
    # Returned content must be valid docx (starts with zip PK header signature)
    assert response.content.startswith(b"PK")


def test_redact_endpoint_rejects_non_docx():
    """Hinglish: Verify /redact endpoint rejects non-docx extension."""
    response = client.post(
        "/redact",
        files={"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Only DOCX files are allowed" in response.json()["detail"]


def test_redact_endpoint_rejects_oversized_file():
    """Hinglish: Verify /redact endpoint rejects files exceeding 15MB limit."""
    huge_data = io.BytesIO(b"0" * (16 * 1024 * 1024))  # 16 MB
    response = client.post(
        "/redact",
        files={"file": ("large.docx", huge_data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_redact_endpoint_rejects_malformed_docx():
    """Hinglish: Verify /redact endpoint rejects malformed/corrupted docx files."""
    corrupted_data = io.BytesIO(b"garbage PK signature but not zip")
    response = client.post(
        "/redact",
        files={"file": ("corrupt.docx", corrupted_data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert response.status_code == 422
    assert "Could not open DOCX file" in response.json()["detail"]
