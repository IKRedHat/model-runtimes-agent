"""Allowlisted `oc` invocation for deployment QA."""

from __future__ import annotations

import logging
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

# Subcommands permitted for automated cluster operations.
ALLOWED_OC_SUBCOMMANDS = frozenset(
    {
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
) -> subprocess.CompletedProcess:
    """
    Run `oc` with an allowlisted subcommand. `args` is the full argv after `oc`
    (e.g. ["get", "ns", "model-validation"]).
    """
    if not args:
        raise ValueError("oc args empty")
    sub = args[0]
    if sub not in ALLOWED_OC_SUBCOMMANDS:
        raise ValueError(f"oc subcommand not allowed: {sub!r}; allowed={sorted(ALLOWED_OC_SUBCOMMANDS)}")

    full = ["oc", *args]
    logger.debug("Running %s", " ".join(full))
    return subprocess.run(
        full,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


__all__ = ["run_oc", "ALLOWED_OC_SUBCOMMANDS"]
