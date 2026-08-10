#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""Scan the project's dependency manifests for known vulnerabilities.

Wraps osv-scanner (https://google.github.io/osv-scanner/) and fails the build on
findings at or above a severity threshold (CRITICAL and HIGH by default).

Findings that cannot be fixed by upgrading are documented in osv-scanner.toml,
which the scanner picks up automatically.

Usage:
    make security-scan                      # gate on CRITICAL + HIGH
    uv run scripts/security_scan.py --fail-on critical
    uv run scripts/security_scan.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OSV_IMAGE = "ghcr.io/google/osv-scanner:latest"

# Manifests that pin the versions we actually install.
SCAN_TARGETS = ["uv.lock", "requirements.txt", "dev-requirements.txt"]

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
SEVERITY_STYLE = {
    "CRITICAL": "bold red",
    "HIGH": "red",
    "MEDIUM": "yellow",
    "LOW": "dim",
    "UNKNOWN": "dim",
}


@dataclass
class Finding:
    source: str
    package: str
    version: str
    severity: str
    score: str
    ids: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)


def bucket(score: str) -> str:
    """Map a CVSS base score onto a qualitative severity rating."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if value >= 9.0:
        return "CRITICAL"
    if value >= 7.0:
        return "HIGH"
    if value >= 4.0:
        return "MEDIUM"
    return "LOW"


def scanner_command(output_path: Path) -> list[str] | None:
    """Return the osv-scanner invocation, preferring a native binary."""
    targets = [t for t in SCAN_TARGETS if (PROJECT_ROOT / t).exists()]
    if not targets:
        return None

    if shutil.which("osv-scanner"):
        return [
            "osv-scanner",
            "scan",
            "source",
            *[str(PROJECT_ROOT / t) for t in targets],
            "--format=json",
            f"--output-file={output_path}",
        ]

    if shutil.which("docker"):
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{PROJECT_ROOT}:/src:ro",
            "-v",
            f"{output_path.parent}:/out",
            OSV_IMAGE,
            "scan",
            "source",
            *[f"/src/{t}" for t in targets],
            "--format=json",
            f"--output-file=/out/{output_path.name}",
        ]

    return None


def parse(report: dict) -> list[Finding]:
    findings: list[Finding] = []
    for result in report.get("results", []):
        source = Path(result.get("source", {}).get("path", "?")).name
        for pkg in result.get("packages", []):
            info = pkg["package"]
            vulns = {v["id"]: v for v in pkg.get("vulnerabilities", [])}
            for group in pkg.get("groups", []):
                score = group.get("max_severity", "")
                fixed: set[str] = set()
                for vuln_id in group.get("ids", []):
                    for affected in vulns.get(vuln_id, {}).get("affected", []):
                        name = affected.get("package", {}).get("name", "")
                        if name.lower() != info["name"].lower():
                            continue
                        for rng in affected.get("ranges", []):
                            for event in rng.get("events", []):
                                if "fixed" in event:
                                    fixed.add(event["fixed"])
                findings.append(
                    Finding(
                        source=source,
                        package=info["name"],
                        version=info["version"],
                        severity=bucket(score),
                        score=score or "-",
                        ids=group.get("ids", []),
                        fixed=sorted(fixed),
                    )
                )
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.package))
    return findings


def render(findings: list[Finding]) -> None:
    if not findings:
        console.print("[green]No known vulnerabilities reported.[/]")
        return

    table = Table(title="osv-scanner findings", show_lines=False)
    table.add_column("Severity")
    table.add_column("Score", justify="right")
    table.add_column("Package")
    table.add_column("Advisory")
    table.add_column("Fixed in")
    table.add_column("Manifest")

    for finding in findings:
        table.add_row(
            f"[{SEVERITY_STYLE[finding.severity]}]{finding.severity}[/]",
            finding.score,
            f"{finding.package} {finding.version}",
            ", ".join(finding.ids),
            ", ".join(finding.fixed) or "-",
            finding.source,
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on",
        default="high",
        choices=["critical", "high", "medium", "low", "never"],
        help="lowest severity that fails the scan (default: high)",
    )
    parser.add_argument("--json", type=Path, help="also write the raw JSON report here")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "osv.json"
        command = scanner_command(report_path)
        if command is None:
            console.print(
                "[red]Neither osv-scanner nor docker is available.[/]\n"
                "Install it from https://google.github.io/osv-scanner/installation/"
            )
            return 2

        console.print(f"[dim]$ {' '.join(command)}[/]")
        # osv-scanner exits non-zero when it finds anything; severity gating is ours.
        subprocess.run(command, cwd=PROJECT_ROOT, check=False)

        if not report_path.exists():
            console.print("[red]osv-scanner produced no report.[/]")
            return 2
        report = json.loads(report_path.read_text())
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))

    findings = parse(report)
    render(findings)

    if args.fail_on == "never":
        return 0

    threshold = SEVERITIES.index(args.fail_on.upper())
    blocking = [f for f in findings if SEVERITIES.index(f.severity) <= threshold]
    if blocking:
        console.print(
            f"\n[bold red]{len(blocking)} finding(s) at or above "
            f"{args.fail_on.upper()}.[/] Upgrade the package, or document why it "
            "cannot be fixed in osv-scanner.toml."
        )
        return 1

    console.print(f"\n[green]No findings at or above {args.fail_on.upper()}.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
