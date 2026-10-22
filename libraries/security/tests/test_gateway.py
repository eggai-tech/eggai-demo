from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from libraries.ml.dspy.language_model import TrackingLM
from libraries.observability.tracing import TracedMessage
from libraries.security.gateway import Run, gateway_token, run_headers, run_var
from libraries.security.handler import authenticate
from libraries.security.keycloak import Keycloak

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture
def run():
    token = run_var.set(Run("gateway-token", TRACEPARENT, "msg-1"))
    yield
    run_var.reset(token)


def _completion(**kwargs):
    return litellm.ModelResponse(
        choices=[{"message": {"content": "hi"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_without_run():
    assert run_headers() == {}
    assert gateway_token() == ""


def test_run_headers_are_trace_and_run_only(run):
    assert run_headers() == {"traceparent": TRACEPARENT, "x-foundry-run": "msg-1"}


def test_llm_call_sends_gateway_token_and_run_headers(run, monkeypatch):
    completion = MagicMock(side_effect=_completion)
    monkeypatch.setattr(litellm, "completion", completion)

    TrackingLM("openai/insurance", cache=False).forward(messages=[{"role": "user", "content": "hello"}])

    kwargs = completion.call_args.kwargs
    assert kwargs["api_key"] == "gateway-token"
    assert {"traceparent": TRACEPARENT, "x-foundry-run": "msg-1"}.items() <= kwargs["headers"].items()


def test_llm_call_without_gateway_token_keeps_environment_key(monkeypatch):
    completion = MagicMock(side_effect=_completion)
    monkeypatch.setattr(litellm, "completion", completion)
    token = run_var.set(Run("", TRACEPARENT, "msg-1"))

    TrackingLM("openai/insurance", cache=False).forward(messages=[{"role": "user", "content": "hello"}])
    run_var.reset(token)

    assert "api_key" not in completion.call_args.kwargs
    assert "x-foundry-run" not in completion.call_args.kwargs["headers"]


class Settings:
    keycloak_url = ""
    keycloak_realm = "insurance"
    keycloak_client_id = "insurance-billing"
    keycloak_client_secret = ""
    gateway_audience = ""


class RecordingKeycloak(Keycloak):
    def __init__(self, settings):
        super().__init__(settings, scope="")
        self.exchanges = []

    async def exchange(self, token: str, audience: str, scope: str) -> str:
        self.exchanges.append((token, audience, scope))
        return f"exchanged-{token}"


def _message():
    return TracedMessage(
        type="billing_request",
        source="Triage",
        data={"message_id": "msg-7", "security_context": {"access_token": "obo"}},
        traceparent=TRACEPARENT,
    )


@pytest.mark.asyncio
async def test_authenticate_exchanges_the_user_token_for_the_gateway():
    class WithGateway(Settings):
        gateway_audience = "foundry-ai"

    keycloak = RecordingKeycloak(WithGateway())
    assert await authenticate(keycloak, _message(), AsyncMock(), "Billing", "conn-1")

    assert keycloak.exchanges == [("obo", "foundry-ai", "foundry-ai-scope insurance-billing-scope")]
    assert run_var.get() == Run("exchanged-obo", TRACEPARENT, "msg-7")
    assert gateway_token() == "exchanged-obo"
    assert run_headers() == {"traceparent": TRACEPARENT, "x-foundry-run": "msg-7"}


@pytest.mark.asyncio
async def test_authenticate_without_gateway_audience_does_not_exchange():
    keycloak = RecordingKeycloak(Settings())
    assert await authenticate(keycloak, _message(), AsyncMock(), "Billing", "conn-1")

    assert keycloak.exchanges == []
    assert gateway_token() == ""
    assert run_var.get() == Run("", TRACEPARENT, "msg-7")
