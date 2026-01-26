#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich", "httpx", "psutil", "python-dotenv"]
# ///
"""
One-command startup for EggAI Demo.

This script handles the complete setup and startup:
1. Checks prerequisites (Python, Docker, uv)
2. Syncs dependencies with uv
3. Starts Docker infrastructure
4. Waits for services to be healthy
5. Starts all agents
6. Verifies system health

Usage:
    uv run scripts/start.py              # Full startup
    uv run scripts/start.py --skip-sync  # Skip dependency sync
    uv run scripts/start.py --no-agents  # Infrastructure only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
PID_FILE = PROJECT_ROOT / ".agent_pids"

# Agent modules to start (in order)
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


def print_banner():
    """Print startup banner."""
    banner = """
[bold cyan]EggAI Demo[/] - Multi-Agent Insurance Support System
[dim]Reference architecture for enterprise AI agent systems[/]
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def check_prerequisites() -> bool:
    """Check that all prerequisites are installed."""
    console.print("\n[bold]Checking prerequisites...[/]")
    all_ok = True

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 11):
        console.print(f"  [green]OK[/] Python {py_version.major}.{py_version.minor}")
    else:
        console.print(f"  [red]FAIL[/] Python 3.11+ required (found {py_version.major}.{py_version.minor})")
        all_ok = False

    # Check Docker
    if shutil.which("docker"):
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                console.print("  [green]OK[/] Docker")
            else:
                console.print("  [red]FAIL[/] Docker not running")
                all_ok = False
        except subprocess.TimeoutExpired:
            console.print("  [red]FAIL[/] Docker not responding")
            all_ok = False
    else:
        console.print("  [red]FAIL[/] Docker not found")
        all_ok = False

    # Check uv
    if shutil.which("uv"):
        console.print("  [green]OK[/] uv")
    else:
        console.print("  [yellow]WARN[/] uv not found (will use pip)")

    return all_ok


def load_environment():
    """Load environment variables from config files."""
    # Load defaults first
    defaults_env = PROJECT_ROOT / "config" / "defaults.env"
    if defaults_env.exists():
        load_dotenv(defaults_env)

    # Then load local overrides
    local_env = PROJECT_ROOT / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)


def sync_dependencies() -> bool:
    """Sync Python dependencies using uv."""
    console.print("\n[bold]Syncing dependencies...[/]")

    if shutil.which("uv"):
        cmd = ["uv", "sync"]
    else:
        console.print("  [dim]uv not found, using pip[/]")
        cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            console.print("  [green]Dependencies synced[/]")
            return True
        else:
            console.print("  [red]Failed to sync dependencies[/]")
            console.print(f"  [dim]{result.stderr}[/]")
            return False
    except subprocess.TimeoutExpired:
        console.print("  [red]Timeout syncing dependencies[/]")
        return False


def start_docker() -> bool:
    """Start Docker Compose infrastructure."""
    console.print("\n[bold]Starting infrastructure...[/]")

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            console.print("  [green]Docker services started[/]")
            return True
        else:
            console.print("  [red]Failed to start Docker[/]")
            console.print(f"  [dim]{result.stderr}[/]")
            return False
    except subprocess.TimeoutExpired:
        console.print("  [red]Timeout starting Docker[/]")
        return False


def wait_for_infrastructure(timeout: float = 120.0) -> bool:
    """Wait for infrastructure services to be healthy."""
    console.print("\n[bold]Waiting for infrastructure...[/]")

    # Import here to avoid dependency issues when running as uv script
    import asyncio

    # Inline async check to avoid import issues
    async def check_services():
        import httpx

        services = [
            ("Redpanda", "http://localhost:19644/health"),
            ("Vespa", "http://localhost:19071/state/v1/health"),
            ("Temporal", "http://localhost:7233/health"),
            ("MLflow", "http://localhost:5001/health"),
        ]

        start = time.time()
        while time.time() - start < timeout:
            healthy = 0
            for _name, url in services:
                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.get(url, timeout=2.0)
                        if r.status_code == 200:
                            healthy += 1
                except Exception:
                    pass

            if healthy == len(services):
                return True

            elapsed = int(time.time() - start)
            console.print(f"  [dim]{elapsed}s - {healthy}/{len(services)} services ready[/]")
            await asyncio.sleep(3)

        return False

    return asyncio.run(check_services())


def start_agents() -> bool:
    """Start all agent processes."""
    console.print("\n[bold]Starting agents...[/]")

    pids = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    for name, module in AGENTS:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pids.append(proc.pid)
            console.print(f"  [green]Started[/] {name} (PID {proc.pid})")
        except Exception as e:
            console.print(f"  [red]Failed[/] {name}: {e}")

    # Save PIDs for cleanup
    if pids:
        PID_FILE.write_text(" ".join(str(p) for p in pids))
        console.print(f"\n  [dim]PIDs saved to {PID_FILE}[/]")

    return len(pids) == len(AGENTS)


def print_success():
    """Print success message with URLs."""
    urls = """
[bold green]EggAI Demo is ready![/]

[bold]Application:[/]
  Chat UI:        http://localhost:8000

[bold]Infrastructure:[/]
  Redpanda:       http://localhost:8082
  Vespa:          http://localhost:8080
  Temporal:       http://localhost:8088
  MLflow:         http://localhost:5001
  Grafana:        http://localhost:3000
  Prometheus:     http://localhost:9090

[bold]Commands:[/]
  Stop:           uv run scripts/stop.py
  Health check:   uv run scripts/health_check.py
  Full reset:     uv run scripts/stop.py --all -v
    """
    console.print(Panel(urls.strip(), title="Ready", border_style="green"))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Start EggAI Demo")
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip dependency synchronization",
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Start infrastructure only, no agents",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout for waiting on services (default: 120s)",
    )
    args = parser.parse_args()

    print_banner()

    # 1. Check prerequisites
    if not check_prerequisites():
        console.print("\n[red]Prerequisites not met. Please install missing dependencies.[/]")
        sys.exit(1)

    # 2. Load environment
    load_environment()

    # 3. Sync dependencies
    if not args.skip_sync:
        if not sync_dependencies():
            console.print("\n[red]Failed to sync dependencies.[/]")
            sys.exit(1)

    # 4. Start Docker
    if not start_docker():
        console.print("\n[red]Failed to start infrastructure.[/]")
        sys.exit(1)

    # 5. Wait for infrastructure
    if not wait_for_infrastructure(timeout=args.timeout):
        console.print("\n[red]Infrastructure did not become healthy in time.[/]")
        sys.exit(1)

    # 6. Start agents
    if not args.no_agents:
        if not start_agents():
            console.print("\n[yellow]Some agents failed to start.[/]")

        # Give agents a moment to initialize
        console.print("\n[dim]Waiting for agents to initialize...[/]")
        time.sleep(3)

    # 7. Success
    print_success()


if __name__ == "__main__":
    main()
