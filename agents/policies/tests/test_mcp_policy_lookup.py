"""The MCP client's bearer is the agent's gateway credential: the same key litellm sends on the agent's model
calls, so the gateway sees one identity per agent at both gates (AIC-AGT-007)."""

from agents.policies.agent.tools.mcp_policy_lookup import gateway_key


def test_openai_provider_uses_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-eggai-policies-0123456789abcdef")
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    assert (
        gateway_key("openai/cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic")
        == "sk-eggai-policies-0123456789abcdef"
    )


def test_lm_studio_provider_uses_lm_studio_api_key(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_API_KEY", "sk-eggai-policies-kind")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert gateway_key("lm_studio/google/gemma-3-12b") == "sk-eggai-policies-kind"


def test_other_providers_follow_litellms_convention(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert gateway_key("anthropic/claude-sonnet-5") == "sk-ant"


def test_missing_key_is_empty_not_an_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert gateway_key("openai/gpt-4o") == ""
