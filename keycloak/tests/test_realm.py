import json
from pathlib import Path

REALM = json.loads((Path(__file__).parent.parent / "realm-export.json").read_text())


def _client(client_id):
    return next(c for c in REALM["clients"] if c["clientId"] == client_id)


def _scope(name):
    return next(s for s in REALM["clientScopes"] if s["name"] == name)


def test_realm_clients():
    assert REALM["realm"] == "insurance"
    assert {c["clientId"] for c in REALM["clients"]} == {
        "insurance-web", "insurance-frontend", "insurance-triage",
        "insurance-policies", "insurance-billing", "insurance-claims", "insurance-escalation",
    }
    assert _client("insurance-web")["publicClient"] is True
    assert _client("insurance-web")["redirectUris"] == ["${INSURANCE_WEB_URL}/*"]
    for exchanging, secret in (("insurance-frontend", "${INSURANCE_FRONTEND_CLIENT_SECRET}"),
                               ("insurance-triage", "${INSURANCE_TRIAGE_CLIENT_SECRET}")):
        client = _client(exchanging)
        assert client["secret"] == secret
        assert client["attributes"]["standard.token.exchange.enabled"] == "true"
        assert "insurance-policies-scope" in client["optionalClientScopes"]
        assert not any(s.endswith("-scope") and s.startswith("insurance-") for s in client["defaultClientScopes"])


def test_agent_scopes_carry_audience_and_scp():
    for agent, scp in (
        ("frontend", "api://insurance-frontend/Chat.ReadWrite"),
        ("triage", "api://insurance-triage/Route.ReadWrite"),
        ("policies", "api://insurance-policies/Policies.ReadWrite"),
        ("billing", "api://insurance-billing/Billing.ReadWrite"),
        ("claims", "api://insurance-claims/Claims.ReadWrite"),
        ("escalation", "api://insurance-escalation/Tickets.ReadWrite"),
    ):
        mappers = {m["protocolMapper"]: m["config"] for m in _scope(f"insurance-{agent}-scope")["protocolMappers"]}
        assert mappers["oidc-audience-mapper"]["included.client.audience"] == f"insurance-{agent}"
        assert mappers["oidc-hardcoded-claim-mapper"]["claim.value"] == scp


def test_users_and_policies():
    users = {u["username"]: u for u in REALM["users"]}
    assert users["john"]["attributes"]["policy_numbers"] == ["A12345"]
    assert users["jane"]["attributes"]["policy_numbers"] == ["B67890"]
    assert users["alice"]["attributes"]["policy_numbers"] == ["C24680"]
    assert users["alice"]["realmRoles"] == ["insurance-admin"]
    assert all(u["credentials"][0]["value"] == "${INSURANCE_USER_PASSWORD}" for u in users.values())
