"""
Legacy post-pass for LLM-authored deployment matrices.

.. warning:: **Test-only legacy module -- not imported by any production code path.**

   Deployability is now computed deterministically by ``deployability_engine`` and
   written by ``deployability_decision``; the HTML report no longer applies this
   reconcile by default.  This module is retained solely so that the existing
   ``tests/test_deployability_reconcile.py`` test suite continues to pass.  It is
   **not** imported or called from any production entry point, CLI command, or
   pipeline stage.

   Do **not** add new production imports of this module.  All GPU-family inference
   and FP8-capability logic should live in ``deployability_engine``.

These helpers remain for backward-compatible tests and optional callers that still
need to flip historical FP8 false negatives when gpu_info.txt proves Hopper/Ada-class
hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# NVIDIA / common tokens that support FP8 (W8A8) per project decision matrix
_FP8_CAPABLE_SUBSTR = (
    "h100",
    "h200",
    "h800",
    "hopper",
    "l4",
    "l40",
    "l40s",
    "ada",
    "b100",
    "b200",
    "gb200",
    "mi300",
    "mi325",
)

# Reasons that should NOT be overridden (capacity / count / unrelated)
_BLOCKERS_SKIP_RECONCILE = (
    "insufficient",
    "not enough",
    "below the inferred",
    "tensor parallel",
    "tensor-parallel",
    "oom",
    "out of memory",
    "no gpu",
    "no accelerators",
    "unreachable",
    "unauthoriz",
    "forbidden",
)


def cluster_supports_fp8_from_gpu_text(gpu_text: str) -> bool:
    """True if gpu_info (or similar) text indicates FP8-capable accelerators."""
    if not gpu_text or not gpu_text.strip():
        return False
    t = gpu_text.lower()
    return any(s in t for s in _FP8_CAPABLE_SUBSTR)


def _mentions_fp8_or_related(reason: str) -> bool:
    r = reason.lower()
    return (
        "fp8" in r
        or "w8a8" in r
        or "8-bit floating" in r
        or ("quantization" in r and "hopper" in r)
    )


def _should_skip_reconcile(reason: str) -> bool:
    r = reason.lower()
    return any(b in r for b in _BLOCKERS_SKIP_RECONCILE)


def _wrong_generation_assumption(reason: str) -> bool:
    """
    Heuristic: LLM assumed Ampere/A100 or 'likely' wrong class while discussing FP8/Hopper.
    """
    r = reason.lower()
    if not _mentions_fp8_or_related(reason):
        return False
    hints = (
        "a100",
        "ampere",
        "likely",
        "probably",
        "may be",
        "misclassif",
        "do not support fp8",
        "without hopper",
        "not hopper",
        "requires hopper",
        "requires nvidia hopper",
        "hardware incompatibility",
    )
    return any(h in r for h in hints)


def reconcile_deployment_matrix_entries(
    matrix: list[dict[str, Any]],
    gpu_text: str,
) -> list[dict[str, Any]]:
    """
    Flip deployable False → True when GPU evidence shows FP8-capable hardware and the
    stored reason looks like a mistaken Ampere-vs-Hopper classification.
    """
    if not cluster_supports_fp8_from_gpu_text(gpu_text):
        return [dict(e) for e in matrix]

    out: list[dict[str, Any]] = []
    for raw in matrix:
        if not isinstance(raw, dict):
            continue
        e = dict(raw)
        reason = str(e.get("reason", ""))
        if e.get("deployable") is False and _mentions_fp8_or_related(reason):
            if not _should_skip_reconcile(reason) and _wrong_generation_assumption(reason):
                e["deployable"] = True
                snippet = gpu_text.strip().splitlines()
                gpu_line = next(
                    (ln for ln in snippet if "gpu product" in ln.lower()),
                    "",
                )
                e["reason"] = (
                    "Reconciled against gpu_info.txt: cluster has FP8-capable GPU(s) "
                    f"({gpu_line.strip() or 'see GPU Product in report'}). "
                    "Original concern assumed incompatible hardware; evidence shows Hopper/Ada-class or compatible AMD."
                )
        out.append(e)
    return out


def reconcile_deployment_matrix_json_file(
    matrix_path: str | Path,
    gpu_text: str,
) -> str | None:
    """Load deployment_matrix.json, reconcile, rewrite; return new JSON string or None.

    ``matrix_path`` is validated before any read: the resolved path must end with a
    file named exactly ``deployment_matrix.json`` so ``..`` segments cannot pivot to
    arbitrary filenames (e.g. ``/etc/passwd``).
    """
    raw = Path(matrix_path).expanduser()
    try:
        resolved = raw.resolve()
    except OSError:
        return None
    if resolved.name != "deployment_matrix.json":
        return None
    if not gpu_text.strip():
        return None
    if not resolved.is_file():
        return None
    try:
        text = resolved.read_text(encoding="utf-8").strip()
        if not text:
            return None
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
    else:
        return None
    fixed = reconcile_deployment_matrix_entries(rows, gpu_text)
    return json.dumps(fixed, indent=2)


__all__ = [
    "cluster_supports_fp8_from_gpu_text",
    "reconcile_deployment_matrix_entries",
    "reconcile_deployment_matrix_json_file",
]
