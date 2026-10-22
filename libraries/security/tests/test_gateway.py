from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from libraries.ml.dspy.language_model import TrackingLM
from libraries.observability.tracing import TracedMessage
from libraries.security.gateway import Run, run_headers, run_var
from libraries.security.handler import authenticate
from libraries.security.keycloak import Keycloak

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture
def run():
    token = run_var.set(Run("user-token", TRACEPARENT, "msg-1"))
    yield
    run_var.reset(token)


def _completion(**kwargs):
    return litellm.ModelResponse(
        choices=[{"message": {"content": "hi"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_run_headers_without_run():
    assert run_headers() == {}


def test_llm_call_reads_token_file_per_call_and_sends_run_headers(run, tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    completion = MagicMock(side_effect=_completion)
    monkeypatch.setattr(litellm, "completion", completion)
    lm = TrackingLM("openai/insurance", cache=False, api_key_file=str(token_file))

    token_file.write_text("sa-token-1\n")
    lm.forward(messages=[{"role": "user", "content": "hello"}])
    token_file.write_text("sa-token-2\n")
    lm.forward(messages=[{"role": "user", "content": "hello"}])

    first, second = (c.kwargs for c in completion.call_args_list)
    assert first["api_key"] == "sa-token-1"
    assert second["api_key"] == "sa-token-2"
    assert {
        "x-foundry-user-token": "user-token",
        "traceparent": TRACEPARENT,
        "x-foundry-run": "msg-1",
    }.items() <= second["headers"].items()


def test_llm_call_without_token_file_keeps_environment_key(run, monkeypatch):
    completion = MagicMock(side_effect=_completion)
    monkeypatch.setattr(litellm, "completion", completion)

    TrackingLM("openai/insurance", cache=False).forward(messages=[{"role": "user", "content": "hello"}])

    assert "api_key" not in completion.call_args.kwargs
    assert "x-foundry-user-token" not in completion.call_args.kwargs["headers"]


class Settings:
    keycloak_url = ""
    keycloak_realm = "insurance"
    keycloak_client_id = "insurance-billing"
    keycloak_client_secret = ""


@pytest.mark.asyncio
async def test_authenticate_binds_the_run():
    msg = TracedMessage(
        type="billing_request",
        source="Triage",
        data={"message_id": "msg-7", "security_context": {"access_token": "obo"}},
        traceparent=TRACEPARENT,
    )
    assert await authenticate(Keycloak(Settings(), scope=""), msg, AsyncMock(), "Billing", "conn-1")
    assert run_var.get() == Run("obo", TRACEPARENT, "msg-7")
