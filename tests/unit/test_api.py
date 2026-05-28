"""Unit tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Health check should return 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_mocked():
    """Ask endpoint should return answer with mocked agent."""
    mock_result = {
        "answer": "Delta Lake is a storage layer.",
        "sources": ["https://docs.databricks.com"],
        "route": "rag",
    }
    with patch("src.api.routes.run_agent", return_value=mock_result):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What is Delta Lake?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Delta Lake is a storage layer."
    assert data["route"] == "rag"


def test_ask_endpoint_empty_question():
    """Empty question should return 422."""
    response = client.post(
        "/api/v1/ask",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_ask_endpoint_missing_field():
    """Missing question field should return 422."""
    response = client.post("/api/v1/ask", json={})
    assert response.status_code == 422