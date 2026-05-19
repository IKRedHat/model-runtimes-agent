"""Shared constants and helpers for the qa_kserve package."""

from __future__ import annotations

import os

_DEFAULT_MAX_GPU_COUNT = 8


def max_gpu_allowed() -> int:
    """Maximum GPU count from QA_MAX_GPU_COUNT env var (default 8)."""
    raw = os.environ.get("QA_MAX_GPU_COUNT", str(_DEFAULT_MAX_GPU_COUNT)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_GPU_COUNT


__all__ = ["max_gpu_allowed"]
