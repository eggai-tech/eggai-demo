from contextlib import asynccontextmanager

import pytest

from agents.policies.agent.tools import mcp_policy_lookup as mod
from libraries.security.gateway import Run, run_var

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.mark.asyncio
async def test_lookup_sends_service_token_and_run_headers_per_call(tmp_path, monkeypatch):
    sent = []

    @asynccontextmanager
    async def client(url, headers=None, httpx_client_factory=None):
        sent.append(headers)
        raise ConnectionError("stop after the headers")
        yield

    monkeypatch.setattr(mod, "streamablehttp_client", client)
    run_var.set(Run("obo-policies", TRACEPARENT, "msg-1"))
    token_file = tmp_path / "token"

    for token in ("sa-token-1", "sa-token-2"):
        token_file.write_text(token)
        with pytest.raises(ConnectionError):
            await mod.lookup("http://gateway/mcp", str(token_file), "A12345")

    assert [h["Authorization"] for h in sent] == ["Bearer sa-token-1", "Bearer sa-token-2"]
    assert sent[1]["x-foundry-user-token"] == "obo-policies"
    assert sent[1]["traceparent"] == TRACEPARENT
    assert sent[1]["x-foundry-run"] == "msg-1"
