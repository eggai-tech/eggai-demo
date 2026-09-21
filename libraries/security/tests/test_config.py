from libraries.core.config import BaseAgentConfig


class Settings(BaseAgentConfig):
    app_name: str = "test"
    prometheus_metrics_port: int = 9999


def test_keycloak_settings_default_off(monkeypatch):
    for name in ("KEYCLOAK_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert s.keycloak_url == ""
    assert s.keycloak_realm == "insurance"
    assert s.keycloak_client_id == ""
    assert s.keycloak_client_secret == ""


def test_every_agent_exposes_keycloak():
    from agents.billing.config import keycloak as billing
    from agents.claims.config import keycloak as claims
    from agents.escalation.config import keycloak as escalation
    from agents.frontend.config import keycloak as frontend
    from agents.policies.agent.config import keycloak as policies
    from agents.triage.config import keycloak as triage

    assert [k.scope for k in (frontend, triage, policies, billing, claims, escalation)] == [
        "api://insurance-frontend/Chat.ReadWrite",
        "api://insurance-triage/Route.ReadWrite",
        "api://insurance-policies/Policies.ReadWrite",
        "api://insurance-billing/Billing.ReadWrite",
        "api://insurance-claims/Claims.ReadWrite",
        "api://insurance-escalation/Tickets.ReadWrite",
    ]
