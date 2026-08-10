#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""Scan first-party source for insecure code patterns.

Wraps opengrep (https://github.com/opengrep/opengrep), the LGPL fork of
semgrep, and fails the build on findings at or above a severity threshold
(CRITICAL and HIGH by default).

Scope mirrors sonar-project.properties: agents/, libraries/, scripts/ and the
CI workflows, minus tests, docs and virtualenvs.

Findings we choose not to fix are suppressed with an inline `# nosemgrep: <rule
id>` comment and documented in opengrep-suppressions.toml. This script enforces
that register: an undocumented suppression, or one whose ignoreUntil date has
passed, fails the scan.

Usage:
    make sast-scan                      # gate on CRITICAL + HIGH
    uv run scripts/sast_scan.py --fail-on medium
    uv run scripts/sast_scan.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPRESSIONS_FILE = PROJECT_ROOT / "opengrep-suppressions.toml"

# Open rule registries. opengrep does not ship semgrep's proprietary Pro rules,
# so these are the community packs; p/default is the union auto-selected by
# `--config auto`, named explicitly here so CI and laptops run the same set.
RULESETS = ["p/default", "p/python", "p/security-audit", "p/secrets"]

# First-party code only - everything else is vendored or generated.
SCAN_TARGETS = ["agents", "libraries", "scripts", ".github"]

# Mirrors sonar.exclusions / sonar.test.inclusions.
EXCLUDES = [
    "tests",
    "test_*.py",
    "*_test.py",
    "shared_test_utils.py",
    "conftest.py",
    "docs",
    ".venv",
    "venv",
    "node_modules",
]

# opengrep reports either the legacy semgrep triple (ERROR/WARNING/INFO) or the
# newer CVSS-style names, depending on the rule. Normalise both.
SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
SEVERITY_ALIASES = {
    "CRITICAL": "CRITICAL",
    "ERROR": "HIGH",
    "HIGH": "HIGH",
    "WARNING": "MEDIUM",
    "MEDIUM": "MEDIUM",
    "INFO": "LOW",
    "LOW": "LOW",
}
SEVERITY_STYLE = {
    "CRITICAL": "bold red",
    "HIGH": "red",
    "MEDIUM": "yellow",
    "LOW": "dim",
    "UNKNOWN": "dim",
}

NOSEMGREP_RE = re.compile(r"#\s*nosemgrep:\s*(?P<ids>[\w.\-, ]+)")


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    message: str


@dataclass
class Suppression:
    rule: str
    path: str
    ignore_until: date
    reason: str


def bucket(severity: str) -> str:
    """Map an opengrep severity label onto a qualitative rating."""
    return SEVERITY_ALIASES.get((severity or "").upper(), "UNKNOWN")


def opengrep_binary() -> str | None:
    """Locate opengrep, including the default install.sh location."""
    found = shutil.which("opengrep")
    if found:
        return found
    fallback = Path.home() / ".opengrep" / "cli" / "latest" / "opengrep"
    return str(fallback) if fallback.is_file() else None


def scanner_command(binary: str, output_path: Path) -> list[str]:
    targets = [t for t in SCAN_TARGETS if (PROJECT_ROOT / t).exists()]
    return [
        binary,
        "scan",
        *[f"--config={c}" for c in RULESETS],
        *[f"--exclude={p}" for p in EXCLUDES],
        "--disable-version-check",
        "--json",
        f"--json-output={output_path}",
        *targets,
    ]


def parse(report: dict) -> list[Finding]:
    findings = [
        Finding(
            path=str(Path(r.get("path", "?"))),
            line=r.get("start", {}).get("line", 0),
            rule=r.get("check_id", "?"),
            severity=bucket(r.get("extra", {}).get("severity", "")),
            message=" ".join(r.get("extra", {}).get("message", "").split()),
        )
        for r in report.get("results", [])
    ]
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.path, f.line))
    return findings


def render(findings: list[Finding]) -> None:
    if not findings:
        console.print("[green]No insecure code patterns reported.[/]")
        return

    table = Table(title="opengrep findings", show_lines=False)
    table.add_column("Severity")
    table.add_column("Location")
    table.add_column("Rule")
    table.add_column("Message")

    for finding in findings:
        table.add_row(
            f"[{SEVERITY_STYLE[finding.severity]}]{finding.severity}[/]",
            f"{finding.path}:{finding.line}",
            finding.rule,
            finding.message[:90],
        )
    console.print(table)


def load_suppressions() -> list[Suppression]:
    if not SUPPRESSIONS_FILE.exists():
        return []
    raw = tomllib.loads(SUPPRESSIONS_FILE.read_text())
    return [
        Suppression(
            rule=entry["id"],
            path=entry["path"],
            ignore_until=entry["ignoreUntil"],
            reason=entry["reason"].strip(),
        )
        for entry in raw.get("Suppressions", [])
    ]


def inline_suppressions() -> set[tuple[str, str]]:
    """Every `# nosemgrep: <rule id>` comment in the scanned tree."""
    found: set[tuple[str, str]] = set()
    for target in SCAN_TARGETS:
        root = PROJECT_ROOT / target
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            if any(part in EXCLUDES for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            for match in NOSEMGREP_RE.finditer(text):
                for rule in match.group("ids").split(","):
                    if rule.strip():
                        found.add((rel, rule.strip()))
    return found


def audit_suppressions(suppressions: list[Suppression]) -> list[str]:
    """Return the reasons the suppression register is not trustworthy."""
    problems: list[str] = []
    today = date.today()
    documented = {(s.path, s.rule) for s in suppressions}
    inline = inline_suppressions()

    for suppression in suppressions:
        if suppression.ignore_until < today:
            problems.append(
                f"expired {suppression.ignore_until}: {suppression.rule} "
                f"in {suppression.path} - re-evaluate and set a new date"
            )
        if (suppression.path, suppression.rule) not in inline:
            problems.append(
                f"stale entry: no `# nosemgrep: {suppression.rule}` comment "
                f"remains in {suppression.path} - delete the entry"
            )

    for path, rule in sorted(inline - documented):
        problems.append(
            f"undocumented: `# nosemgrep: {rule}` in {path} has no entry in "
            f"{SUPPRESSIONS_FILE.name}"
        )
    return problems


def render_suppressions(suppressions: list[Suppression]) -> None:
    if not suppressions:
        return
    table = Table(title="active suppressions", show_lines=False)
    table.add_column("Expires")
    table.add_column("Location")
    table.add_column("Rule")
    for suppression in suppressions:
        table.add_row(
            str(suppression.ignore_until), suppression.path, suppression.rule
        )
    console.print(table)


def run_scan(args: argparse.Namespace) -> dict | None:
    binary = opengrep_binary()
    if binary is None:
        console.print(
            "[red]opengrep is not installed.[/]\n"
            "Install it with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep"
            "/main/install.sh | bash"
        )
        return None

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "opengrep.json"
        command = scanner_command(binary, report_path)
        console.print(f"[dim]$ {' '.join(command)}[/]")
        # opengrep only exits non-zero with --error; severity gating is ours.
        subprocess.run(command, cwd=PROJECT_ROOT, check=False)

        if not report_path.exists():
            console.print("[red]opengrep produced no report.[/]")
            return None
        report = json.loads(report_path.read_text())

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return report


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

    if not os.environ.get("SEMGREP_SEND_METRICS"):
        os.environ["SEMGREP_SEND_METRICS"] = "off"

    report = run_scan(args)
    if report is None:
        return 2

    findings = parse(report)
    render(findings)

    suppressions = load_suppressions()
    render_suppressions(suppressions)
    problems = audit_suppressions(suppressions)
    if problems:
        console.print("\n[bold red]Suppression register is out of date:[/]")
        for problem in problems:
            console.print(f"  - {problem}")
        return 1

    if args.fail_on == "never":
        return 0

    threshold = SEVERITIES.index(args.fail_on.upper())
    blocking = [f for f in findings if SEVERITIES.index(f.severity) <= threshold]
    if blocking:
        console.print(
            f"\n[bold red]{len(blocking)} finding(s) at or above "
            f"{args.fail_on.upper()}.[/] Fix the code, or add an inline "
            f"`# nosemgrep: <rule id>` and document why in "
            f"{SUPPRESSIONS_FILE.name}."
        )
        return 1

    console.print(f"\n[green]No findings at or above {args.fail_on.upper()}.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
