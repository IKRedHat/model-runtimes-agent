"""Classify pod / ISVC failures for self-heal decisions."""

from __future__ import annotations

import json
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


def logs_hint_oom(log_text: str) -> bool:
    """Lightweight check for OOM patterns in logs when status is unclear."""
    lower = log_text.lower()
    return any(
        x in lower
        for x in (
            "out of memory",
            "oom",
            "cuda out of memory",
            "killed process",
            "signal 9",
        )
    )


def summarize_tail(text: str, max_chars: int = 4000) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


__all__ = ["FailureKind", "classify_pod_json", "logs_hint_oom", "summarize_tail"]
