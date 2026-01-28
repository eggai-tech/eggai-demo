#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "rich"]
# ///

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    url: str
    message: str = ""
    latency_ms: float = 0


INFRASTRUCTURE_SERVICES: list[tuple[str, str, Callable | None]] = [
    ("Redpanda", "http://localhost:19644/v1/status/ready", None),
    ("Redpanda Console", "http://localhost:8082/", None),
    ("Vespa Config", "http://localhost:19071/state/v1/health", None),
    ("Vespa Query", "http://localhost:8080/", None),
    ("Temporal UI", "http://localhost:8081/", None),
    ("MLflow", "http://localhost:5001/health", None),
    ("MinIO", "http://localhost:9001/", None),
    ("Grafana", "http://localhost:3000/api/health", None),
    ("Prometheus", "http://localhost:9090/-/healthy", None),
]

AGENT_SERVICES: list[tuple[str, str, Callable | None]] = [
    ("Frontend", "http://localhost:8000/", None),
]


async def check_http_service(
    name: str,
    url: str,
    validator: Callable[[dict], bool] | None = None,
    timeout: float = 5.0,
) -> ServiceHealth:
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout, follow_redirects=True)
            latency = (time.perf_counter() - start) * 1000

            if response.status_code in (200, 301, 302):
                if validator:
                    try:
                        data = response.json()
                        if validator(data):
                            return ServiceHealth(
                                name, HealthStatus.HEALTHY, url, latency_ms=latency
                            )
                        else:
                            return ServiceHealth(
                                name,
                                HealthStatus.UNHEALTHY,
                                url,
                                "Validation failed",
                                latency,
                            )
                    except Exception:
                        return ServiceHealth(
                            name, HealthStatus.HEALTHY, url, latency_ms=latency
                        )
                return ServiceHealth(name, HealthStatus.HEALTHY, url, latency_ms=latency)
            else:
                return ServiceHealth(
                    name,
                    HealthStatus.UNHEALTHY,
                    url,
                    f"HTTP {response.status_code}",
                    latency,
                )

    except httpx.ConnectError:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name, HealthStatus.UNHEALTHY, url, "Connection refused", latency
        )
    except httpx.TimeoutException:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(name, HealthStatus.UNHEALTHY, url, "Timeout", latency)
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name, HealthStatus.UNHEALTHY, url, str(e)[:50], latency
        )


async def check_all_services(
    include_agents: bool = True,
) -> list[ServiceHealth]:
    services = INFRASTRUCTURE_SERVICES.copy()
    if include_agents:
        services.extend(AGENT_SERVICES)

    tasks = [check_http_service(name, url, validator) for name, url, validator in services]
    return await asyncio.gather(*tasks)


async def wait_for_services(
    timeout: float = 120.0,
    poll_interval: float = 2.0,
    include_agents: bool = False,
) -> bool:
    console.print(f"[yellow]Waiting for services (timeout: {timeout}s)...[/]")

    start = time.time()
    while time.time() - start < timeout:
        results = await check_all_services(include_agents=include_agents)
        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        total = len(results)

        if healthy_count == total:
            console.print(f"[green]All {total} services healthy![/]")
            return True

        unhealthy = [r.name for r in results if r.status != HealthStatus.HEALTHY]
        elapsed = int(time.time() - start)
        console.print(
            f"[dim]{elapsed}s[/] Waiting for: {', '.join(unhealthy)} ({healthy_count}/{total})"
        )

        await asyncio.sleep(poll_interval)

    console.print("[red]Timeout waiting for services![/]")
    return False


def print_results(results: list[ServiceHealth]) -> bool:
    table = Table(title="Service Health Check")
    table.add_column("Status", style="bold", width=3)
    table.add_column("Service", style="cyan")
    table.add_column("URL", style="dim")
    table.add_column("Latency", justify="right")
    table.add_column("Message")

    all_healthy = True
    for r in results:
        if r.status == HealthStatus.HEALTHY:
            status_icon = "[green]OK[/]"
        else:
            status_icon = "[red]FAIL[/]"
            all_healthy = False

        latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "-"

        table.add_row(
            status_icon,
            r.name,
            r.url,
            latency_str,
            r.message,
        )

    console.print(table)

    if all_healthy:
        console.print("\n[bold green]All services healthy![/]")
    else:
        console.print("\n[bold red]Some services unhealthy![/]")

    return all_healthy


async def verify_all(include_agents: bool = True) -> bool:
    results = await check_all_services(include_agents=include_agents)
    return print_results(results)


def main():
    parser = argparse.ArgumentParser(description="Check EggAI Demo service health")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until all services are healthy",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for --wait mode (default: 120)",
    )
    parser.add_argument(
        "--include-agents",
        action="store_true",
        help="Include agent services in health check",
    )
    args = parser.parse_args()

    if args.wait:
        success = asyncio.run(
            wait_for_services(
                timeout=args.timeout,
                include_agents=args.include_agents,
            )
        )
    else:
        success = asyncio.run(verify_all(include_agents=args.include_agents))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
