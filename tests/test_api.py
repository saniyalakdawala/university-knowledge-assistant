import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    """Test GET /api/health returns valid schema and status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "group4" in data
    assert "group9" in data
    assert data["group4"]["status"] == "ready"
    assert data["group4"]["chunks_count"] > 0
    # Ensure no secrets leak
    assert "api_key" not in str(data).lower()
    assert "secret" not in str(data).lower()

def test_static_frontend_routes():
    """Test static frontend files are served under root."""
    # Index HTML
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "University Knowledge Assistant" in res_index.text

    # CSS
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert "var(--primary-red)" in res_css.text

    # JS
    res_js = client.get("/script.js")
    assert res_js.status_code == 200
    assert "DOMContentLoaded" in res_js.text

    # NMIMS Logo Asset
    res_logo = client.get("/assets/nmims_logo.png")
    assert res_logo.status_code == 200
    assert res_logo.headers["content-type"] == "image/png"

def test_invalid_chat_request():
    """Test validation errors on empty or invalid bodies."""
    # Empty body
    res_empty = client.post("/api/chat", json={})
    assert res_empty.status_code == 422

    # Empty question
    res_blank = client.post("/api/chat", json={"question": "   "})
    assert res_blank.status_code == 422
