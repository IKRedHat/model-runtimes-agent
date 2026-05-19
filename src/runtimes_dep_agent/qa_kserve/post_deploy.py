"""Post-deploy smoke inference, scale-to-zero, and namespace cleanup."""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .oc_cli import run_oc

logger = logging.getLogger(__name__)

# Blocked for post-deploy HTTP client (SSRF): RFC1918, loopback, metadata, ULA, etc.
_V4_SSRF_NETWORKS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("0.0.0.0/8"),
)
_V6_SSRF_NETWORKS: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::ffff:0:0/96"),
)

# Kubernetes and cloud metadata hostnames that must never be contacted by
# post-deploy smoke tests — prevents SSRF to cluster-internal services.
_BLOCKED_K8S_HOSTNAMES: frozenset[str] = frozenset({
    "kubernetes",
    "kubernetes.default",
    "kubernetes.default.svc",
    "kubernetes.default.svc.cluster.local",
    "metadata.google.internal",
})


def _ipv4_ssrf_blocked(addr: ipaddress.IPv4Address) -> bool:
    return any(addr in net for net in _V4_SSRF_NETWORKS)


def _ipv6_ssrf_blocked(addr: ipaddress.IPv6Address) -> bool:
    if any(addr in net for net in _V6_SSRF_NETWORKS):
        return True
    mapped = addr.ipv4_mapped
    if mapped is not None:
        return _ipv4_ssrf_blocked(mapped)
    return False


def _ip_ssrf_blocked(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if addr.version == 4:
        return _ipv4_ssrf_blocked(addr)
    return _ipv6_ssrf_blocked(addr)


def _inference_url_ssrf_block_reason(url: str) -> str | None:
    """
    Return a human-readable block reason, or None if ``url`` is http(s) and
    resolved addresses are not in blocked ranges.
    """
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return f"Blocked non-HTTP(S) scheme: {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return "Blocked URL: missing hostname"

    # Block known K8s internal and cloud metadata hostnames
    if host in _BLOCKED_K8S_HOSTNAMES or host.endswith(".svc.cluster.local") or host.endswith(".svc"):
        return f"Blocked: K8s/cloud internal hostname {host!r}"

    offenders: list[str] = []
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(5.0)
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        return f"Blocked: DNS resolution failed for {host!r}: {e}"
    finally:
        socket.setdefaulttimeout(old_timeout)

    if not infos:
        return f"Blocked: no DNS results for {host!r}"

    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_s = sockaddr[0]
        if not isinstance(ip_s, str) or ip_s in seen:
            continue
        seen.add(ip_s)
        try:
            addr = ipaddress.ip_address(ip_s)
        except ValueError:
            return f"Blocked: invalid resolved address {ip_s!r}"
        if _ip_ssrf_blocked(addr):
            offenders.append(addr.compressed)

    if offenders:
        uniq = ", ".join(dict.fromkeys(offenders))
        return (
            f"Blocked SSRF-risk: host {host!r} resolves to forbidden address(es): {uniq}"
        )
    return None


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
                return _normalize_base_url(url, log)
            comps = st.get("components") or {}
            pred = comps.get("predictor")
            if isinstance(pred, dict):
                u = (pred.get("url") or "").strip()
                if u:
                    return _normalize_base_url(u, log)
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
        cand = candidates[0][1].rstrip("/")
        reason = _inference_url_ssrf_block_reason(cand)
        if reason:
            _append(log, reason)
            return ""
        return cand
    return ""


def _normalize_base_url(url: str, log: list[str]) -> str:
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        norm = u.rstrip("/")
    else:
        norm = f"https://{u}".rstrip("/")
    reason = _inference_url_ssrf_block_reason(norm)
    if reason:
        _append(log, reason)
        return ""
    return norm


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

    Before any network I/O, ``base_url`` is checked by ``_inference_url_ssrf_block_reason``
    (http/https only, DNS resolution, blocked private/link-local/metadata-style addresses).
    On failure this returns ``(False, "<Blocked …>")`` with the same message string used
    when resolving inference URLs.

    TLS: verified against the default trust store unless ``tls_ca_file`` is set
    (``ssl.create_default_context(cafile=...)``) or ``tls_insecure`` is True
    (explicit dev-only; disables verification).
    """
    block = _inference_url_ssrf_block_reason(base_url.rstrip("/"))
    if block:
        return False, block

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
