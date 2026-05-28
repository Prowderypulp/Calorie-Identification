"""Tests for the FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_analyze_no_file(client):
    resp = client.post("/api/v1/analyze")
    assert resp.status_code == 422  # missing required file


def test_analyze_invalid_content_type(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_empty_file(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_analyze_valid_image(client):
    """Create a minimal valid JPEG-like image and send it."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(200, 100, 50)).save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("food.jpg", buf, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "food_items" in data
    assert "total_nutrients" in data
    assert "processing_time_ms" in data
