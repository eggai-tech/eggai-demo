import pytest
from fastapi.testclient import TestClient

from agents.claims.api_main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "claims-agent-api",
        "version": "1.0.0"
    }


def test_list_claims(client):
    response = client.get("/api/v1/claims?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    assert "total" in data
    assert isinstance(data["claims"], list)


def test_list_claims_with_filters(client):
    response = client.get("/api/v1/claims?status=In Review&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    # Check all returned claims have the correct status
    for claim in data["claims"]:
        assert claim["status"] == "In Review"


def test_get_claim_by_number(client):
    response = client.get("/api/v1/claims/1001")
    assert response.status_code == 200
    data = response.json()
    assert data["claim_number"] == "1001"


def test_get_nonexistent_claim(client):
    response = client.get("/api/v1/claims/9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_claims_statistics(client):
    response = client.get("/api/v1/claims/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_claims" in data
    assert "total_estimated" in data
    assert "by_status" in data
    assert "by_policy" in data
    assert "average_estimate" in data
