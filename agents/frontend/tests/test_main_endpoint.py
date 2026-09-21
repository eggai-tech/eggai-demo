import pytest
from fastapi.testclient import TestClient

from ..main import api, settings


@pytest.fixture(autouse=True)
def temp_public_dir(tmp_path, monkeypatch):
    """Monkey-patch settings.public_dir to a temporary directory."""
    monkeypatch.setattr(settings, "public_dir", str(tmp_path))
    return tmp_path


def test_read_root_success(temp_public_dir):
    html_file = temp_public_dir / "index.html"
    content = "<html><body>OK</body></html>"
    html_file.write_text(content, encoding="utf-8")
    client = TestClient(api)
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == content


def test_read_root_not_found(temp_public_dir):
    client = TestClient(api)
    response = client.get("/")
    assert response.status_code == 404


def test_read_root_error(temp_public_dir, monkeypatch):
    html_file = temp_public_dir / "index.html"
    html_file.write_text("data", encoding="utf-8")
    # Simulate aiofiles.open() throwing an unexpected error
    import aiofiles
    monkeypatch.setattr(
        aiofiles,
        'open',
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("fail")),
    )
    client = TestClient(api)
    response = client.get("/")
    assert response.status_code == 500
    assert "An error occurred" in response.json().get("detail", "")



def test_config_defaults():
    client = TestClient(api)
    body = client.get("/config").json()
    assert {"name": "Grafana", "url": "http://localhost:3000"} in body["platformLinks"]
    assert len(body["platformLinks"]) == 7
    assert list(body) == ["platformLinks", "keycloak"]


def test_config_from_settings(monkeypatch):
    monkeypatch.setattr(
        settings,
        "platform_links",
        "Grafana=http://grafana.eggai.localhost, Redpanda=http://redpanda.eggai.localhost",
    )
    client = TestClient(api)
    body = client.get("/config").json()
    assert body["platformLinks"] == [
        {"name": "Grafana", "url": "http://grafana.eggai.localhost"},
        {"name": "Redpanda", "url": "http://redpanda.eggai.localhost"},
    ]


@pytest.fixture
def upstream(monkeypatch):
    import httpx

    from .. import main

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(201, json={"echo": request.url.path})

    monkeypatch.setattr(main, "upstream", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


def test_api_proxy_forwards_get_with_query(upstream):
    client = TestClient(api)
    response = client.get("/api/claims/claims?limit=1")
    assert response.status_code == 201
    assert response.json() == {"echo": "/api/v1/claims"}
    assert str(upstream[0].url) == "http://localhost:8003/api/v1/claims?limit=1"


def test_api_proxy_forwards_post_body(upstream):
    client = TestClient(api)
    client.post("/api/policies/kb/search/vector", json={"query": "roof"})
    request = upstream[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:8002/api/v1/kb/search/vector"
    assert request.content == b'{"query":"roof"}'
    assert request.headers["content-type"] == "application/json"


def test_api_proxy_unknown_agent(upstream):
    client = TestClient(api)
    assert client.get("/api/audit/anything").status_code == 404
    assert upstream == []
