"""KServe-based deployment QA (replaces ODH container tests)."""

from typing import Any

__all__ = ["run_kserve_deployment_qa"]


def __getattr__(name: str) -> Any:
    if name == "run_kserve_deployment_qa":
        from .pipeline import run_kserve_deployment_qa

        return run_kserve_deployment_qa
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
