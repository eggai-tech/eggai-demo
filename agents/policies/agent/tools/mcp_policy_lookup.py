"""Personal policy lookup over MCP through Agent Router, used instead of the in-process tool when
POLICIES_MCP_URL is set. One short MCP session per call (initialize, tools/list, tools/call, close) so
the call runs on its own event loop inside dspy's worker thread. A refusal by the gateway's policy
decision point (HTTP 403 with a JSON body) becomes the tool's answer: the mcp client
itself only reports "Connection closed" for a refused POST, so the body is captured on the way in."""

import asyncio
from collections.abc import Callable

import dspy
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from libraries.observability.logger import get_console_logger
from libraries.security.gateway import run_headers, service_token
from libraries.security.keycloak import assert_policy_access

logger = get_console_logger("policies_agent.tools.mcp")

TOOL_NAME = (
    "policies-mcp__get_policy_details"  # Agent Router prefixes every tool with the backendRef name
)


async def lookup(url: str, token_file: str, policy_number: str) -> str:
    refusals: list[str] = []

    async def capture(response: httpx.Response) -> None:
        if response.status_code >= 400:
            await response.aread()
            refusals.append(f"HTTP {response.status_code} {response.text}")

    def client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers, timeout=timeout, auth=auth, event_hooks={"response": [capture]}
        )

    # The same credentials the agent's model calls carry: its ServiceAccount token and the run.
    headers = {"Authorization": f"Bearer {service_token(token_file)}", **run_headers()}
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


def mcp_policy_lookup(url: str, token_file: str) -> Callable[[str], str]:
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
        return asyncio.run(lookup(url, token_file, policy_number))

    return get_personal_policy_details
