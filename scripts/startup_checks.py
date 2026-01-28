from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

PROJECT_ROOT = Path(__file__).parent.parent

console = Console()


def check_prerequisites() -> bool:
    console.print("\n[bold]Checking prerequisites...[/]")
    all_ok = True

    py_version = sys.version_info
    if py_version >= (3, 11):
        console.print(f"  [green]OK[/] Python {py_version.major}.{py_version.minor}")
    else:
        console.print(f"  [red]FAIL[/] Python 3.11+ required (found {py_version.major}.{py_version.minor})")
        all_ok = False

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

    if shutil.which("uv"):
        console.print("  [green]OK[/] uv")
    else:
        console.print("  [yellow]WARN[/] uv not found (will use pip)")

    return all_ok


def load_environment():
    defaults_env = PROJECT_ROOT / "config" / "defaults.env"
    if defaults_env.exists():
        load_dotenv(defaults_env)

    local_env = PROJECT_ROOT / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)


def sync_dependencies() -> bool:
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
