"""Post-deploy smoke inference, scale-to-zero, and namespace cleanup."""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .oc_cli import run_oc

logger = logging.getLogger(__name__)


def resolve_inference_base_url(isvc_name: str, namespace: str, log: list[str]) -> str:
    """
    Best-effort external/base URL for OpenAI-compatible inference (scheme + host, no path).
    Tries InferenceService status, then OpenShift Routes in the namespace.
    """
    r = run_oc(
        ["get", "inferenceservice", isvc_name, "-n", namespace, "-o", "json"],
        timeout=60,
    )
    if r.returncode == 0 and r.stdout:
        try:
            doc = json.loads(r.stdout)
            st = doc.get("status") or {}
            url = (st.get("url") or "").strip()
            if url:
                return _normalize_base_url(url)
            comps = st.get("components") or {}
            pred = comps.get("predictor")
            if isinstance(pred, dict):
                u = (pred.get("url") or "").strip()
                if u:
                    return _normalize_base_url(u)
        except json.JSONDecodeError:
            pass

    rr = run_oc(["get", "routes", "-n", namespace, "-o", "json"], timeout=90)
    if rr.returncode != 0 or not rr.stdout:
        _append(log, f"(routes) no routes or error: {rr.stderr or ''}")
        return ""
    try:
        rd = json.loads(rr.stdout)
    except json.JSONDecodeError:
        return ""
    items = rd.get("items") or []
    # Prefer route whose name matches or contains the InferenceService name
    candidates: list[tuple[int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("metadata") or {}).get("name") or ""
        host = (item.get("spec") or {}).get("host") or ""
        if not host:
            continue
        score = 0
        if name == isvc_name:
            score = 100
        elif isvc_name in name:
            score = 50
        elif name.startswith(isvc_name):
            score = 40
        else:
            score = 1
        tls = item.get("spec", {}).get("tls")
        scheme = "https" if tls else "http"
        candidates.append((score, f"{scheme}://{host}"))
    candidates.sort(key=lambda x: -x[0])
    if candidates:
        return candidates[0][1]
    return ""


def _normalize_base_url(url: str) -> str:
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u.rstrip("/")
    return f"https://{u}".rstrip("/")


def _append(log: list[str], msg: str) -> None:
    log.append(msg)
    print(f"[QA] {msg}", flush=True)


def _smoke_ssl_context(
    *,
    tls_ca_file: str | None,
    tls_insecure: bool,
    log: list[str],
) -> ssl.SSLContext | None:
    """
    SSL context for smoke HTTPS requests.

    Default (both flags off / no CA file): ``None`` so ``urlopen`` uses the
    interpreter default context (hostname + cert verification enabled).

    ``QA_SMOKE_TLS_INSECURE`` (``tls_insecure``): dev/test only — disables verification
    without calling ``ssl._create_unverified_context()``.

    ``QA_SMOKE_TLS_CA_FILE`` (``tls_ca_file``): optional PEM bundle path.
    """
    if tls_insecure:
        _append(
            log,
            "smoke TLS: insecure mode (QA_SMOKE_TLS_INSECURE) — cert verification disabled",
        )
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if tls_ca_file:
        p = Path(tls_ca_file).expanduser()
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        if resolved.is_file():
            return ssl.create_default_context(cafile=str(resolved))
        _append(
            log,
            f"smoke TLS: CA file missing at {tls_ca_file!r}, using default trust store",
        )
    return None


def post_chat_completions_smoke(
    base_url: str,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    timeout_s: float,
    log: list[str],
    tls_ca_file: str | None = None,
    tls_insecure: bool = False,
) -> tuple[bool, str]:
    """
    POST /v1/chat/completions (OpenAI-compatible). Returns (ok, detail_or_response_snippet).

    TLS: verified against the default trust store unless ``tls_ca_file`` is set
    (``ssl.create_default_context(cafile=...)``) or ``tls_insecure`` is True
    (explicit dev-only; disables verification).
    """
def post_chat_completions_smoke(
    base_url: str,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    timeout_s: float,
    verify_tls: bool,
    log: list[str],
) -> tuple[bool, str]:
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    blocked_hosts = {
        "169.254.169.254", "metadata.google.internal",
        "localhost", "127.0.0.1", "[::1]",
    }
    if parsed.hostname in blocked_hosts:
        return False, f"Blocked internal/metadata host: {parsed.hostname}"
    if parsed.hostname and (
        parsed.hostname.startswith("10.") or
        parsed.hostname.startswith("192.168.") or
        parsed.hostname.startswith("172.16.")
    ):
        return False, f"Blocked private network: {parsed.hostname}"
    
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    ctx = _smoke_ssl_context(tls_ca_file=tls_ca_file, tls_insecure=tls_insecure, log=log)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            code = resp.getcode()
            if code != 200:
                return False, f"HTTP {code}: {body[:500]}"
            # Light validation: parse JSON and look for choices/content
            try:
                obj = json.loads(body)
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    _append(log, f"smoke inference OK ({len(body)} bytes response)")
                    return True, body[:800]
            except json.JSONDecodeError:
                pass
            return True, body[:800]
    except urllib.error.HTTPError as e:
        err_body = (e.read() or b"").decode("utf-8", errors="replace")
        return False, f"HTTPError {e.code}: {err_body[:600]}"
    except Exception as e:
        logger.exception("smoke inference failed")
        return False, str(e)[:500]


def patch_isvc_scale_to_zero(isvc_name: str, namespace: str, log: list[str]) -> bool:
    """Set predictor minReplicas (and maxReplicas when supported) to 0 (best-effort)."""
    patches = (
        {"spec": {"predictor": {"minReplicas": 0, "maxReplicas": 0}}},
        {"spec": {"predictor": {"minReplicas": 0}}},
    )
    last_err = ""
    for p in patches:
        merge_patch = json.dumps(p)
        r = run_oc(
            [
                "patch",
                "inferenceservice",
                isvc_name,
                "-n",
                namespace,
                "--type",
                "merge",
                "-p",
                merge_patch,
            ],
            timeout=120,
        )
        if r.returncode == 0:
            _append(log, f"scaled {isvc_name} with patch {merge_patch}")
            return True
        last_err = r.stderr or r.stdout or ""
    _append(log, f"warn scale-to-zero patch: {last_err}")
    return False


def delete_namespace(namespace: str, log: list[str]) -> bool:
    r = run_oc(["delete", "namespace", namespace, "--wait=true"], timeout=600)
    if r.returncode != 0:
        _append(log, f"QA_ERROR:NAMESPACE_DELETE_FAILED {r.stderr or r.stdout}")
        return False
    _append(log, f"deleted namespace {namespace}")
    return True


__all__ = [
    "delete_namespace",
    "patch_isvc_scale_to_zero",
    "post_chat_completions_smoke",
    "resolve_inference_base_url",
]
