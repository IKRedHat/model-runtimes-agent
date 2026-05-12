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
    # Podman only: True if the engine responds (e.g. `podman info` succeeds).
    # None means not applicable (oc, skopeo) or tool not installed.
    running: bool | None = None
    running_detail: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "path": self.path,
        }
        if self.running is not None:
            d["running"] = self.running
        if self.running_detail:
            d["running_detail"] = self.running_detail
        return d


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


def _podman_engine_running(podman_bin: str = "podman") -> tuple[bool, str]:
    """Return (ok, detail) after probing whether the Podman engine is reachable."""
    try:
        result = subprocess.run(
            [podman_bin, "info"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout or "").strip()
        first = err.splitlines()[0] if err else "podman info failed"
        return False, first[:200]
    except FileNotFoundError:
        return False, "podman not found"
    except subprocess.TimeoutExpired:
        return False, "podman info timed out (engine may be hung or not started)"
    except OSError as exc:
        return False, str(exc)


def check_tool(name: str) -> ToolStatus:
    path = shutil.which(name)
    if not path:
        return ToolStatus(name=name, installed=False)
    version = _get_tool_version(name)
    if not version:
        return ToolStatus(name=name, installed=False, path=path)
    if name != "podman":
        return ToolStatus(name=name, installed=True, version=version, path=path)

    ok, detail = _podman_engine_running(path)
    return ToolStatus(
        name=name,
        installed=True,
        version=version,
        path=path,
        running=ok,
        running_detail=detail if not ok else "",
    )


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
    for r in results:
        if not r.installed:
            return False
        if r.name == "podman" and r.running is False:
            return False
    return True


def _print_tool_status(status: ToolStatus) -> None:
    label = f"  {status.name:<12}"
    if status.installed:
        detail = f"({status.version})" if status.version else ""
        if status.path:
            detail = f"({status.path})" if not detail else f"({status.version}, {status.path})"
        if status.name == "podman" and status.running is False:
            extra = status.running_detail or "engine not reachable"
            print(f"{label}{YELLOW}{CHECK} installed{RESET}  {detail}")
            print(f"{' ' * len(label)}{RED}{CROSS} engine not running{RESET}  {DIM}{extra}{RESET}")
        else:
            print(f"{label}{GREEN}{CHECK} installed{RESET}  {detail}")
            if status.name == "podman" and status.running is True:
                print(f"{' ' * len(label)}{GREEN}{CHECK} engine reachable{RESET}")
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
