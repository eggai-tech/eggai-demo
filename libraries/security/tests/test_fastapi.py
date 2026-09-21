import jwt
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from libraries.security.fastapi import require_bearer
from libraries.security.keycloak import Caller, Keycloak


class Settings:
    keycloak_url = "http://kc:8080"
    keycloak_realm = "insurance"
    keycloak_client_id = "insurance-billing"
    keycloak_client_secret = ""


def _app(keycloak):
    app = FastAPI(dependencies=[Depends(require_bearer(keycloak))])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/v1/billing")
    def billing():
        return {"records": []}

    return app


def test_health_is_open():
    assert TestClient(_app(Keycloak(Settings(), scope="s"))).get("/health").status_code == 200


def test_missing_bearer_is_401():
    assert TestClient(_app(Keycloak(Settings(), scope="s"))).get("/api/v1/billing").status_code == 401


def test_invalid_bearer_is_401(monkeypatch):
    keycloak = Keycloak(Settings(), scope="s")

    def bad(token):
        raise jwt.InvalidTokenError("nope")

    monkeypatch.setattr(keycloak, "validate", bad)
    response = TestClient(_app(keycloak)).get("/api/v1/billing", headers={"Authorization": "Bearer x"})
    assert response.status_code == 401


def test_valid_bearer_passes(monkeypatch):
    keycloak = Keycloak(Settings(), scope="s")
    monkeypatch.setattr(keycloak, "validate", lambda token: Caller("alice", "Alice", [], ["insurance-admin"]))
    response = TestClient(_app(keycloak)).get("/api/v1/billing", headers={"Authorization": "Bearer x"})
    assert response.status_code == 200


def test_disabled_keycloak_passes():
    class Off(Settings):
        keycloak_url = ""

    assert TestClient(_app(Keycloak(Off(), scope="s"))).get("/api/v1/billing").status_code == 200
