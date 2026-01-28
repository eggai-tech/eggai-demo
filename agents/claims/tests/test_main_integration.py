"""Integration tests for Claims main module"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

with patch('agents.claims.main.create_kafka_transport'):
    with patch('agents.claims.main.eggai_set_default_transport'):
        with patch('agents.claims.main.claims_agent') as mock_agent:
            mock_agent.start = AsyncMock()
            mock_agent.stop = MagicMock()

            with patch('agents.claims.main.init_telemetry'):
                with patch('agents.claims.main.dspy_set_language_model'):
                    with patch('agents.claims.dspy_modules.claims.load_optimized_prompts'):
                        from agents.claims.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_claims_list_endpoint(client):
    response = client.get("/api/v1/claims?limit=5")
    assert response.status_code == 200


def test_claims_stats_endpoint(client):
    response = client.get("/api/v1/claims/stats")
    assert response.status_code == 200


def test_app_has_cors_middleware(client):
    # Make a preflight request
    response = client.options(
        "/api/v1/claims",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    # Check CORS headers are present (middleware is configured)
    assert "access-control-allow-methods" in response.headers
