from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

PROJECT_ROOT = Path(__file__).parent.parent
PID_FILE = PROJECT_ROOT / ".agent_pids"

console = Console()

AGENTS = [
    ("Frontend", "agents.frontend.main"),
    ("Triage", "agents.triage.main"),
    ("Billing", "agents.billing.main"),
    ("Claims", "agents.claims.main"),
    ("Policies", "agents.policies.agent.main"),
    ("Escalation", "agents.escalation.main"),
    ("Audit", "agents.audit.main"),
    ("Document Ingestion", "agents.policies.ingestion.start_worker"),
]


def start_agents() -> bool:
    console.print("\n[bold]Starting agents...[/]")

    pids = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    use_uv = shutil.which("uv") is not None

    for name, module in AGENTS:
        try:
            if use_uv:
                cmd = ["uv", "run", "python", "-m", module]
            else:
                cmd = [sys.executable, "-m", module]

            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pids.append(proc.pid)
            console.print(f"  [green]Started[/] {name} (PID {proc.pid})")
        except Exception as e:
            console.print(f"  [red]Failed[/] {name}: {e}")

    if pids:
        PID_FILE.write_text(" ".join(str(p) for p in pids))
        console.print(f"\n  [dim]PIDs saved to {PID_FILE}[/]")

    return len(pids) == len(AGENTS)
