"""Classify pod / ISVC failures for self-heal decisions."""

from __future__ import annotations

import json
import re
from typing import Literal

FailureKind = Literal["none", "oom", "image_pull", "crashloop", "pending", "unknown"]


def classify_pod_json(pods_json_stdout: str) -> tuple[FailureKind, str]:
    """
    Inspect `oc get pods ... -o json` output for terminal failure signals.
    """
    try:
        doc = json.loads(pods_json_stdout)
    except json.JSONDecodeError:
        return "unknown", "invalid pod json"

    items = doc.get("items") or []
    messages: list[str] = []

    for pod in items:
        for cs in pod.get("status", {}).get("containerStatuses") or []:
            name = cs.get("name", "?")
            term = cs.get("state", {}).get("terminated") or {}
            if term.get("reason") == "OOMKilled":
                return "oom", f"container {name}: OOMKilled"
            wait = cs.get("state", {}).get("waiting") or {}
            reason = wait.get("reason") or ""
            if reason in ("ImagePullBackOff", "ErrImagePull"):
                return "image_pull", f"container {name}: {reason}"
            if reason == "CrashLoopBackOff":
                messages.append(f"container {name}: CrashLoopBackOff")

        for cs in pod.get("status", {}).get("initContainerStatuses") or []:
            term = cs.get("state", {}).get("terminated") or {}
            if term.get("reason") == "OOMKilled":
                return "oom", f"init {cs.get('name')}: OOMKilled"
            wait = cs.get("state", {}).get("waiting") or {}
            reason = wait.get("reason") or ""
            if reason in ("ImagePullBackOff", "ErrImagePull"):
                return "image_pull", f"init {cs.get('name')}: {reason}"

    phase = ""
    for pod in items:
        phase = pod.get("status", {}).get("phase") or ""
        if phase == "Pending":
            return "pending", "pod Pending"

    if messages:
        return "crashloop", "; ".join(messages)

    return "none", ""


_BASE64ISH_LINE = re.compile(r"^[A-Za-z0-9+/=]{80,}$")


def _oom_killed_from_pod_json(pods_json_stdout: str | None) -> bool:
    """True if any container/initContainer has terminated OOMKilled or exitCode 137."""
    if not pods_json_stdout or not pods_json_stdout.strip():
        return False
    try:
        doc = json.loads(pods_json_stdout)
    except json.JSONDecodeError:
        return False
    items = doc.get("items")
    if not isinstance(items, list):
        return False
    for pod in items:
        if not isinstance(pod, dict):
            continue
        status = pod.get("status")
        if not isinstance(status, dict):
            continue
        for key in ("containerStatuses", "initContainerStatuses"):
            for cs in status.get(key) or []:
                if not isinstance(cs, dict):
                    continue
                state = cs.get("state")
                if not isinstance(state, dict):
                    continue
                term = state.get("terminated")
                if not isinstance(term, dict):
                    continue
                if term.get("reason") == "OOMKilled":
                    return True
                if term.get("exitCode") == 137:
                    return True
    return False


def _log_lines_for_scan(log_text: str) -> str:
    """Drop lines that are likely base64 / opaque blobs to reduce false positives."""
    kept: list[str] = []
    for line in log_text.splitlines():
        s = line.strip()
        if len(s) >= 80 and _BASE64ISH_LINE.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


# Log fallback: word-boundary / phrase patterns only (best-effort; prefer pod JSON).
_LOG_OOM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z0-9])out\s+of\s+memory(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])cuda\s+out\s+of\s+memory(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])torch\.cuda\.outofmemoryerror(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])outofmemoryerror(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])oomkilled(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])\bkilled\s+process\b", re.IGNORECASE),
    re.compile(
        r"(?<![A-Za-z0-9])(?:exit(?:ed)?\s+code\s*137|exited\s+with\s+code\s*137|signal\s*[:#]?\s*9)\b",
        re.IGNORECASE,
    ),
)


def logs_hint_oom(log_text: str, *, pods_json_stdout: str | None = None) -> bool:
    """
    Best-effort OOM hint from pod status and/or container logs.

    When ``pods_json_stdout`` is set (e.g. ``oc get pods -o json``), this prefers
    structured signals: ``pod.status.containerStatuses[].state.terminated.reason ==
    "OOMKilled"`` or ``exitCode == 137`` (and the same for ``initContainerStatuses``).

    Log matching is intentionally conservative (phrases / bounded tokens, base64-like
    lines skipped). For authoritative classification, use pod JSON and
    ``classify_pod_json`` / ``containerStatuses[].state.terminated.reason``.
    """
    if pods_json_stdout is not None and _oom_killed_from_pod_json(pods_json_stdout):
        return True
    if not log_text or not log_text.strip():
        return False
    scan = _log_lines_for_scan(log_text)
    return any(p.search(scan) for p in _LOG_OOM_PATTERNS)


def summarize_tail(text: str, max_chars: int = 4000) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


__all__ = ["FailureKind", "classify_pod_json", "logs_hint_oom", "summarize_tail"]
