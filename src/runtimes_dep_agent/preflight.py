"""Pre-flight dependency checks and colored progress logging."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Sequence


REQUIRED_TOOLS = ("oc", "skopeo")

GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CHECK = "\u2713"
CROSS = "\u2717"


@dataclass
class ToolStatus:
    name: str
    installed: bool
    version: str = ""
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "path": self.path,
        }


def _get_tool_version(name: str) -> str:
    """Best-effort version string for a CLI tool."""
    # oc needs --client to avoid contacting the cluster
    flag_sequences = [["--version"], ["version"]]
    if name == "oc":
        flag_sequences = [["version", "--client"], ["--version"]]

    for flags in flag_sequences:
        try:
            result = subprocess.run(
                [name, *flags],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                continue
            output = (result.stdout or result.stderr).strip()
            first_line = output.splitlines()[0] if output else ""
            if first_line:
                return first_line
        except Exception:
            continue
    return ""


def check_tool(name: str) -> ToolStatus:
    path = shutil.which(name)
    if not path:
        return ToolStatus(name=name, installed=False)
    version = _get_tool_version(name)
    if not version:
        return ToolStatus(name=name, installed=False, path=path)
    return ToolStatus(name=name, installed=True, version=version, path=path)


def run_preflight_checks(
    tools: Sequence[str] = REQUIRED_TOOLS,
    *,
    quiet: bool = False,
) -> list[ToolStatus]:
    """Check that each tool is installed and optionally print status."""
    results: list[ToolStatus] = []
    if not quiet:
        print(f"\n{BOLD}=== Pre-flight Checks ==={RESET}")

    for name in tools:
        status = check_tool(name)
        results.append(status)
        if not quiet:
            _print_tool_status(status)

    if not quiet:
        print()
    return results


def preflight_ok(results: list[ToolStatus]) -> bool:
    return all(r.installed for r in results)


def _print_tool_status(status: ToolStatus) -> None:
    label = f"  {status.name:<12}"
    if status.installed:
        detail = f"({status.version})" if status.version else ""
        if status.path:
            detail = f"({status.path})" if not detail else f"({status.version}, {status.path})"
        print(f"{label}{GREEN}{CHECK} installed{RESET}  {detail}")
    else:
        print(f"{label}{RED}{CROSS} NOT FOUND{RESET}")


# ---------------------------------------------------------------------------
# Progress logger for structured step-by-step terminal output
# ---------------------------------------------------------------------------

@dataclass
class ProgressLogger:
    """Prints numbered, coloured step markers to the terminal."""

    total_steps: int = 5
    _current: int = field(default=0, init=False)

    def section(self, title: str) -> None:
        print(f"\n{BOLD}=== {title} ==={RESET}")

    def step(self, label: str) -> None:
        self._current += 1
        print(f"  {CYAN}[{self._current}/{self.total_steps}]{RESET} {label}")

    def detail(self, msg: str) -> None:
        print(f"        {msg}")

    def success(self, msg: str) -> None:
        print(f"        {GREEN}{CHECK}{RESET} {msg}")

    def fail(self, msg: str) -> None:
        print(f"        {RED}{CROSS}{RESET} {msg}")

    def done(self, elapsed_seconds: float) -> None:
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        if minutes:
            ts = f"{minutes}m {seconds}s"
        else:
            ts = f"{seconds}s"
        print(f"\n{BOLD}=== Completed in {ts} ==={RESET}\n")
