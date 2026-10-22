from contextlib import asynccontextmanager

import pytest

from agents.policies.agent.tools import mcp_policy_lookup as mod
from libraries.security.gateway import Run, run_var

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.mark.asyncio
async def test_lookup_sends_gateway_token_and_run_headers(monkeypatch):
    sent = []

    @asynccontextmanager
    async def client(url, headers=None, httpx_client_factory=None):
        sent.append(headers)
        raise ConnectionError("stop after the headers")
        yield

    monkeypatch.setattr(mod, "streamablehttp_client", client)
    token = run_var.set(Run("gateway-token", TRACEPARENT, "msg-1"))

    with pytest.raises(ConnectionError):
        await mod.lookup("http://gateway/mcp", "A12345")
    run_var.reset(token)

    assert sent == [
        {"Authorization": "Bearer gateway-token", "traceparent": TRACEPARENT, "x-foundry-run": "msg-1"}
    ]
