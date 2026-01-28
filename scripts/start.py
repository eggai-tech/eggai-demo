#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich", "httpx", "psutil", "python-dotenv"]
# ///

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console

from scripts.startup_agents import start_agents
from scripts.startup_checks import check_prerequisites, load_environment, sync_dependencies
from scripts.startup_infra import start_docker, wait_for_infrastructure
from scripts.startup_ui import print_banner, print_success

console = Console()


def main():
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

    if not check_prerequisites():
        console.print("\n[red]Prerequisites not met. Please install missing dependencies.[/]")
        sys.exit(1)

    load_environment()

    if not args.skip_sync:
        if not sync_dependencies():
            console.print("\n[red]Failed to sync dependencies.[/]")
            sys.exit(1)

    if not start_docker():
        console.print("\n[red]Failed to start infrastructure.[/]")
        sys.exit(1)

    if not wait_for_infrastructure(timeout=args.timeout):
        console.print("\n[red]Infrastructure did not become healthy in time.[/]")
        sys.exit(1)

    if not args.no_agents:
        if not start_agents():
            console.print("\n[yellow]Some agents failed to start.[/]")

        console.print("\n[dim]Waiting for agents to initialize...[/]")
        time.sleep(3)

    print_success()


if __name__ == "__main__":
    main()
