"""What an agent presents to the AI gateway on every model and tool call: a Keycloak token for the
gateway's audience, exchanged once per run from the user's token, so it names the agent (azp) and the
user it acts for; and the run (trace, user message id)."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    gateway_token: str
    traceparent: str
    run_id: str


run_var: ContextVar[Run | None] = ContextVar("run", default=None)


def gateway_token() -> str:
    run = run_var.get()
    return run.gateway_token if run else ""


def run_headers() -> dict[str, str]:
    run = run_var.get()
    if run is None:
        return {}
    return {"traceparent": run.traceparent, "x-foundry-run": run.run_id}
