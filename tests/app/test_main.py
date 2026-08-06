"""Tests for the FastAPI application (app/main.py)"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_query_returns_expected_shape():
    response = client.post("/query", json={"query": "What is the total on the Acme invoice?"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"answer", "intent", "sources"}


def test_query_rejects_missing_query_field():
    # Pydantic validation should reject a malformed request body
    # before it ever reaches answer_query() — FastAPI returns 422
    # Unprocessable Entity for validation failures.
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_audio_endpoint_404s_for_missing_file():
    response = client.get("/audio/nonexistent_file.mp3")
    assert response.status_code == 404