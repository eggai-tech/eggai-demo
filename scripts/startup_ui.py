from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console()


def print_banner():
    banner = """
[bold cyan]EggAI Demo[/] - Multi-Agent Insurance Support System
[dim]Reference architecture for enterprise AI agent systems[/]
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def print_success():
    urls = """
[bold green]EggAI Demo is ready![/]

[bold]Application:[/]
  Chat UI:        http://localhost:8000

[bold]Infrastructure:[/]
  Redpanda:       http://localhost:8082
  Vespa:          http://localhost:8080
  Temporal:       http://localhost:8081
  MLflow:         http://localhost:5001
  Grafana:        http://localhost:3000
  Prometheus:     http://localhost:9090

[bold]Commands:[/]
  Stop:           uv run scripts/stop.py
  Health check:   uv run scripts/health_check.py
  Full reset:     uv run scripts/stop.py --all -v
    """
    console.print(Panel(urls.strip(), title="Ready", border_style="green"))
