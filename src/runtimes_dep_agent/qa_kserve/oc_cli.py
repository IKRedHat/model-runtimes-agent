"""Allowlisted `oc` invocation for deployment QA."""

from __future__ import annotations

import logging
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

# Subcommands permitted for automated cluster operations.
ALLOWED_OC_SUBCOMMANDS = frozenset(
    {
        "auth",
        "get",
        "apply",
        "create",
        "delete",
        "patch",
        "logs",
        "wait",
        "project",
        "describe",
        "whoami",
        "version",
    }
)


def run_oc(
    args: Sequence[str],
    *,
    timeout: float | None = 300,
    check: bool = False,
    stdin_input: str | bytes | None = None,
) -> subprocess.CompletedProcess:
    """
    Run `oc` with an allowlisted subcommand. `args` is the full argv after `oc`
    (e.g. ["get", "ns", "model-validation"]).

    When ``stdin_input`` is set, it is written to the child stdin (e.g. ``oc apply -f -``).
    """
    if not args:
        raise ValueError("oc args empty")
    sub = args[0]
    if sub not in ALLOWED_OC_SUBCOMMANDS:
        raise ValueError(f"oc subcommand not allowed: {sub!r}; allowed={sorted(ALLOWED_OC_SUBCOMMANDS)}")

    full = ["oc", *args]
    logger.debug("Running %s", " ".join(full))
    if stdin_input is None:
        return subprocess.run(
            full,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    return subprocess.run(
        full,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        input=stdin_input,
    )


__all__ = ["run_oc", "ALLOWED_OC_SUBCOMMANDS"]
