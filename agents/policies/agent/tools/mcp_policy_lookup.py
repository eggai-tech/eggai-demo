"""Personal policy lookup over MCP through Agent Router, used instead of the in-process tool when
POLICIES_MCP_URL is set. One short MCP session per call (initialize, tools/list, tools/call, close) so
the call runs on its own event loop inside dspy's worker thread. A refusal by the gateway's policy
decision point (OPA ext_authz, HTTP 403 with a JSON body) becomes the tool's answer: the mcp client
itself only reports "Connection closed" for a refused POST, so the body is captured on the way in."""

import asyncio
import os
from collections.abc import Callable

import dspy
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from libraries.observability.logger import get_console_logger
from libraries.security.keycloak import assert_policy_access

logger = get_console_logger("policies_agent.tools.mcp")

TOOL_NAME = (
    "policies-mcp__get_policy_details"  # Agent Router prefixes every tool with the backendRef name
)

# litellm reads the provider's key from the environment by convention (<PROVIDER>_API_KEY); these two
# are the names it uses for the providers the demo runs on.
PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY", "lm_studio": "LM_STUDIO_API_KEY"}


def gateway_key(language_model: str) -> str:
    """The agent's gateway credential: the key litellm sends on this agent's model calls, read from the
    environment variable of the model's provider (the prefix of LANGUAGE_MODEL). The gateway's policy
    decision point maps it to the agent (AIC-AGT-007), at the model gate and at the MCP gate alike."""
    provider = language_model.partition("/")[0]
    return os.environ.get(PROVIDER_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY"), "")


async def lookup(url: str, policy_number: str, key: str) -> str:
    refusals: list[str] = []

    async def capture(response: httpx.Response) -> None:
        if response.status_code >= 400:
            await response.aread()
            refusals.append(f"HTTP {response.status_code} {response.text}")

    def client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers, timeout=timeout, auth=auth, event_hooks={"response": [capture]}
        )

    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with streamablehttp_client(url, headers=headers, httpx_client_factory=client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool = next(t for t in (await session.list_tools()).tools if t.name == TOOL_NAME)
                return await dspy.Tool.from_mcp_tool(session, tool).acall(
                    policy_number=policy_number
                )
    except Exception:
        if not refusals:
            raise
        logger.error(f"MCP tool call refused by the gateway: {refusals[0]}")
        return f"Tool call refused by the policy gateway: {refusals[0]}"


def mcp_policy_lookup(url: str, language_model: str) -> Callable[[str], str]:
    key = gateway_key(language_model)

    def get_personal_policy_details(policy_number: str) -> str:
        """
        Retrieve specific policy details from database using policy number.
        Use this when user provides a policy number and wants their personal policy information.
        Returns JSON with policy data or error message.
        """
        denied = assert_policy_access(policy_number)
        if denied:
            return denied
        logger.info(f"Looking up policy '{policy_number}' over MCP at {url}")
        return asyncio.run(lookup(url, policy_number, key))

    return get_personal_policy_details
