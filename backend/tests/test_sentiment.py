import pytest
from fastapi.testclient import TestClient
from ..app.main import app

client = TestClient(app)

def test_analyze_sentiment():
    response = client.post("/api/v1/analyze", json={"texts": ["I love this!", "This is terrible"]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
