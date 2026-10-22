"""What an agent presents to the AI gateway on every model and tool call: its projected ServiceAccount
token as the bearer, read from the file on each call because the kubelet rotates it, and the run it
acts in (the user's exchanged token, the trace, the user message id)."""

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run:
    user_token: str
    traceparent: str
    run_id: str


run_var: ContextVar[Run | None] = ContextVar("run", default=None)


def service_token(token_file: str) -> str:
    return Path(token_file).read_text().strip()


def run_headers() -> dict[str, str]:
    run = run_var.get()
    if run is None:
        return {}
    return {
        "x-foundry-user-token": run.user_token,
        "traceparent": run.traceparent,
        "x-foundry-run": run.run_id,
    }
