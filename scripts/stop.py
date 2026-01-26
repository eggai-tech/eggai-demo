#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich", "psutil"]
# ///
"""
Graceful shutdown for EggAI Demo.

Stops all running agent processes and optionally Docker services.

Usage:
    uv run scripts/stop.py              # Stop agents only
    uv run scripts/stop.py --all        # Stop agents and Docker
    uv run scripts/stop.py --docker     # Stop Docker only
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil
from rich.console import Console

console = Console()

# Patterns to identify agent processes
AGENT_PATTERNS = [
    "agents.frontend.main",
    "agents.triage.main",
    "agents.billing.main",
    "agents.claims.main",
    "agents.policies.agent.main",
    "agents.policies.ingestion",
    "agents.escalation.main",
    "agents.audit.main",
    "start_worker",
]

PROJECT_ROOT = Path(__file__).parent.parent
PID_FILE = PROJECT_ROOT / ".agent_pids"


def find_agent_processes() -> list[psutil.Process]:
    """Find all running agent processes."""
    agent_processes = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)

            for pattern in AGENT_PATTERNS:
                if pattern in cmdline_str:
                    agent_processes.append(proc)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return agent_processes


def stop_agents() -> int:
    """Stop all running agent processes. Returns count of stopped processes."""
    console.print("[yellow]Stopping agent processes...[/]")

    # First try to read PIDs from file
    stopped_count = 0
    if PID_FILE.exists():
        try:
            pids = PID_FILE.read_text().strip().split()
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                    console.print(f"  Sent SIGTERM to PID {pid}")
                    stopped_count += 1
                except (ProcessLookupError, ValueError):
                    pass
            PID_FILE.unlink()
        except Exception as e:
            console.print(f"[dim]Could not read PID file: {e}[/]")

    # Also find and kill by pattern
    processes = find_agent_processes()
    for proc in processes:
        try:
            console.print(f"  Stopping {proc.name()} (PID {proc.pid})")
            proc.terminate()
            stopped_count += 1
        except psutil.NoSuchProcess:
            pass

    # Wait for graceful shutdown
    if processes:
        gone, alive = psutil.wait_procs(processes, timeout=5)

        # Force kill any remaining
        for proc in alive:
            try:
                console.print(f"  [red]Force killing {proc.name()} (PID {proc.pid})[/]")
                proc.kill()
            except psutil.NoSuchProcess:
                pass

    if stopped_count > 0:
        console.print(f"[green]Stopped {stopped_count} agent process(es)[/]")
    else:
        console.print("[dim]No agent processes found[/]")

    return stopped_count


def stop_docker(remove_volumes: bool = False) -> bool:
    """Stop Docker Compose services."""
    console.print("[yellow]Stopping Docker services...[/]")

    cmd = ["docker", "compose", "down"]
    if remove_volumes:
        cmd.append("-v")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]Docker services stopped[/]")
            return True
        else:
            console.print(f"[red]Failed to stop Docker: {result.stderr}[/]")
            return False
    except FileNotFoundError:
        console.print("[red]Docker not found[/]")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Stop EggAI Demo services")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stop both agents and Docker services",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Stop Docker services only",
    )
    parser.add_argument(
        "--remove-volumes",
        "-v",
        action="store_true",
        help="Remove Docker volumes (full reset)",
    )
    args = parser.parse_args()

    success = True

    if args.docker:
        # Docker only
        success = stop_docker(remove_volumes=args.remove_volumes)
    elif args.all:
        # Both
        stop_agents()
        success = stop_docker(remove_volumes=args.remove_volumes)
    else:
        # Agents only (default)
        stop_agents()

    console.print("\n[bold]Shutdown complete.[/]")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
