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


def start_agents_foreground() -> None:
    """Start all agents in foreground with visible output. Blocks until Ctrl+C."""
    import signal

    console.print("\n[bold]Starting agents in foreground mode...[/]")
    console.print("[dim]Press Ctrl+C to stop all agents[/]\n")

    processes: list[subprocess.Popen] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    use_uv = shutil.which("uv") is not None

    def cleanup(signum=None, frame=None):
        console.print("\n[yellow]Stopping all agents...[/]")
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        console.print("[green]All agents stopped.[/]")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

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
            )
            processes.append(proc)
            console.print(f"  [green]Started[/] {name} (PID {proc.pid})")
        except Exception as e:
            console.print(f"  [red]Failed[/] {name}: {e}")

    if processes:
        pids = [p.pid for p in processes]
        PID_FILE.write_text(" ".join(str(p) for p in pids))

    console.print(f"\n[bold green]All agents running. Logs will appear below.[/]\n")
    console.print("[dim]" + "=" * 60 + "[/]\n")

    # Wait for any process to exit
    while processes:
        for proc in processes[:]:
            ret = proc.poll()
            if ret is not None:
                # Find agent name
                agent_name = "Unknown"
                for name, module in AGENTS:
                    if module in " ".join(proc.args or []):
                        agent_name = name
                        break
                console.print(f"\n[yellow]{agent_name} exited with code {ret}[/]")
                processes.remove(proc)
        import time
        time.sleep(0.5)
