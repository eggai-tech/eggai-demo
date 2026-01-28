from __future__ import annotations

import subprocess
import time
from pathlib import Path

from rich.console import Console

PROJECT_ROOT = Path(__file__).parent.parent

console = Console()


def start_docker() -> bool:
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
    console.print("\n[bold]Waiting for infrastructure...[/]")

    import asyncio

    async def check_services():
        import httpx

        services = [
            ("Redpanda", "http://localhost:19644/v1/status/ready"),
            ("Vespa", "http://localhost:19071/state/v1/health"),
            ("Temporal UI", "http://localhost:8081"),
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
