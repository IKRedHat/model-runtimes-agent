"""Render KServe manifests from deployment-yamls templates."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from pathlib import Path

import yaml

from ..utils.path_utils import detect_repo_root

_K8S_NAME_MAX = 63


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced `{...}` slice (string-aware); avoids trailing junk like `|extra`."""
    t = text.strip()
    start = t.find("{")
    if start < 0:
        return t
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(t)):
        c = t[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return t[start:]


def _parse_dockerconfig_json(text: str) -> tuple[dict, bool]:
    """
    Parse .dockerconfigjson body; return (dict, repaired) where repaired means we dropped trailing junk.
    """
    t = text.strip()
    repaired = False
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, repaired
    except json.JSONDecodeError:
        pass
    snippet = _extract_first_json_object(t)
    obj = json.loads(snippet)
    if not isinstance(obj, dict):
        raise ValueError("dockerconfigjson must be a JSON object")
    if snippet.strip() != t.strip():
        repaired = True
    return obj, repaired


def _b64_decode_loose(raw_b64: str) -> bytes:
    s = re.sub(r"\s+", "", raw_b64)
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    return base64.b64decode(s, validate=False)


def sanitize_k8s_name(raw: str, max_len: int = _K8S_NAME_MAX) -> str:
    """RFC-ish DNS label for InferenceService metadata.name."""
    s = raw.lower().replace("_", "-")
    s = re.sub(r"[^a-z0-9-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-") or "model"
    if s[0].isdigit():
        s = "m-" + s
    return s[:max_len].strip("-")


def normalize_dockerconfig_b64(raw: str) -> str:
    """
    Normalize user input to a single base64-encoded `.dockerconfigjson` Secret data value.

    Accepts: raw JSON object, or base64 of that JSON. Strips accidental suffixes (e.g. ``|`` or
    extra text after the closing ``}``) that cause ``invalid character '|' after top-level value``.
    If the value looks like several base64 chunks separated by ``|``, uses the first chunk only.
    """
    t = raw.strip()
    # Pipe often appears when two payloads or shell noise are concatenated.
    if "|" in t:
        t = t.split("|", 1)[0].strip()

    if t.startswith("{"):
        obj, _repaired = _parse_dockerconfig_json(t)
        canon = json.dumps(obj, separators=(",", ":"), sort_keys=False)
        return base64.b64encode(canon.encode("utf-8")).decode("ascii")

    try:
        dec = _b64_decode_loose(t).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return t.strip()

    try:
        obj, _repaired = _parse_dockerconfig_json(dec)
    except (json.JSONDecodeError, ValueError):
        return t.strip()

    canon = json.dumps(obj, separators=(",", ":"), sort_keys=False)
    return base64.b64encode(canon.encode("utf-8")).decode("ascii")


def build_registry_secret_yaml(
    *,
    registry_host: str,
    dockerconfigjson_b64: str,
    secret_name: str,
    namespace: str,
) -> str:
    """
    Build a kubernetes.io/dockerconfigjson Secret manifest aligned with
    deployment-yamls/oci-secret.yaml (ODH connection annotations + fields).
    """
    obj = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": secret_name,
            "namespace": namespace,
            "annotations": {
                "opendatahub.io/connection-type-protocol": "oci",
                "opendatahub.io/connection-type-ref": "oci-v1",
                "openshift.io/display-name": secret_name,
                "registry.host/openshift": registry_host,
            },
            "labels": {"opendatahub.io/dashboard": "true"},
        },
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": dockerconfigjson_b64.strip(),
            # ODH connection secret shape (matches deployment-yamls/oci-secret.yaml)
            "ACCESS_TYPE": "WyJQdWxsIl0=",  # base64 of ["Pull"]
            "OCI_HOST": base64.b64encode(registry_host.encode("utf-8")).decode("ascii"),
        },
    }
    return yaml.dump(obj, sort_keys=False, default_flow_style=False)


def format_args_block(args: list[str]) -> str:
    """YAML snippet for model.args under predictor.model."""
    if not args:
        return ""
    lines: list[str] = ["      args:"]
    for a in args:
        safe = json.dumps(a)
        lines.append(f"        - {safe}")
    return "\n".join(lines) + "\n"


def format_image_pull_secrets_block(secret_name: str) -> str:
    return f"    imagePullSecrets:\n      - name: {secret_name}\n"


def format_gpu_requests_line(gpu_count: int) -> str:
    if gpu_count <= 0:
        return ""
    return f'        nvidia.com/gpu: "{gpu_count}"\n'


def format_gpu_limits_line(gpu_count: int) -> str:
    if gpu_count <= 0:
        return ""
    return f'          nvidia.com/gpu: "{gpu_count}"\n'


# Memory calculation defaults for pick_cpu_memory.
# VRAM multiplier: accounts for KV cache and runtime overhead beyond raw model weights.
_DEFAULT_MEMORY_MULTIPLIER = 1.25
# Fixed headroom in GiB for GPU drivers and framework overhead.
_DEFAULT_MEMORY_OVERHEAD_GI = 2
# Per-heal-retry memory increment in GiB (progressive resource escalation).
_DEFAULT_HEAL_BUMP_GI = 4
# Minimum base memory in GiB regardless of model size.
_DEFAULT_BASE_MEMORY_GI = 8


def pick_cpu_memory(
    *,
    required_vram_gb: float | None,
    heal_bump: int,
) -> tuple[str, str, str, str]:
    """
    Return cpu_request, memory_request, cpu_limit, memory_limit as Kubernetes quantities.
    heal_bump increases memory tiers on retry. Calculation constants are configurable
    via QA_MEMORY_MULTIPLIER, QA_MEMORY_OVERHEAD_GI, QA_MEMORY_HEAL_BUMP_GI,
    and QA_BASE_MEMORY_GI environment variables.
    """
    try:
        multiplier = float(os.environ.get("QA_MEMORY_MULTIPLIER", str(_DEFAULT_MEMORY_MULTIPLIER)))
    except ValueError:
        multiplier = _DEFAULT_MEMORY_MULTIPLIER
    try:
        overhead_gi = int(os.environ.get("QA_MEMORY_OVERHEAD_GI", str(_DEFAULT_MEMORY_OVERHEAD_GI)))
    except ValueError:
        overhead_gi = _DEFAULT_MEMORY_OVERHEAD_GI
    try:
        heal_gi = int(os.environ.get("QA_MEMORY_HEAL_BUMP_GI", str(_DEFAULT_HEAL_BUMP_GI)))
    except ValueError:
        heal_gi = _DEFAULT_HEAL_BUMP_GI
    try:
        base_min = int(os.environ.get("QA_BASE_MEMORY_GI", str(_DEFAULT_BASE_MEMORY_GI)))
    except ValueError:
        base_min = _DEFAULT_BASE_MEMORY_GI

    base_mem = base_min
    if required_vram_gb is not None and required_vram_gb > 0:
        base_mem = max(base_mem, int(required_vram_gb * multiplier) + overhead_gi + heal_bump * heal_gi)
    else:
        base_mem = base_mem + heal_bump * heal_gi

    mem_req = f"{base_mem}Gi"
    mem_lim = f"{max(base_mem * 2, base_mem + 8)}Gi"
    cpu_req = "2"
    cpu_lim = "8"
    return cpu_req, mem_req, cpu_lim, mem_lim


def bump_memory_quantity(mem: str, factor: float = 1.5) -> str:
    m = re.match(r"^(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti)$", mem.strip())
    if not m:
        return mem
    val, unit = float(m.group(1)), m.group(2)
    nv = val * factor
    if nv >= 1.0:
        return f"{int(round(nv))}{unit}"
    return f"{nv:.1f}{unit}".replace(".0Gi", "Gi")


def halve_max_model_len_args(args: list[str]) -> list[str]:
    out: list[str] = []
    for a in args:
        if a.startswith("--max-model-len="):
            try:
                v = int(a.split("=", 1)[1])
                out.append(f"--max-model-len={max(512, v // 2)}")
            except ValueError:
                out.append(a)
        else:
            out.append(a)
    return out


def render_inference_service(
    *,
    template_text: str,
    isvc_name: str,
    model_image: str,
    vllm_runtime_image: str,
    serving_runtime_name: str,
    model_format: str,
    args: list[str],
    oci_secret_name: str,
    cpu_request: str,
    memory_request: str,
    cpu_limit: str,
    memory_limit: str,
    gpu_count: int,
) -> str:
    """Replace placeholders in inference-service.yaml.template."""
    args_block = format_args_block(args)
    ips_block = format_image_pull_secrets_block(oci_secret_name)
    gpu_line = format_gpu_requests_line(gpu_count)
    gpu_limits_line = format_gpu_limits_line(gpu_count)

    text = template_text
    repl = {
        "__ISVC_NAME__": isvc_name,
        "__MODEL_IMAGE__": model_image,
        "__VLLM_RUNTIME_IMAGE__": vllm_runtime_image,
        "__SERVING_RUNTIME_NAME__": serving_runtime_name,
        "__MODEL_FORMAT__": model_format,
        "__ARGS_BLOCK__": args_block,
        "__IMAGE_PULL_SECRETS_BLOCK__": ips_block,
        "__CPU_REQUEST__": cpu_request,
        "__MEMORY_REQUEST__": memory_request,
        "__CPU_LIMIT__": cpu_limit,
        "__MEMORY_LIMIT__": memory_limit,
        "__GPU_REQUESTS_LINE__": gpu_line.rstrip("\n"),
        "__GPU_LIMITS_LINE__": gpu_limits_line.rstrip("\n"),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def load_inference_template(repo_root: Path | None = None) -> str:
    root = repo_root or detect_repo_root()
    path = root / "deployment-yamls" / "inference-service.yaml.template"
    if not path.exists():
        raise FileNotFoundError(f"InferenceService template missing: {path}")
    return path.read_text(encoding="utf-8")


def load_serving_runtime_template(repo_root: Path | None = None) -> str:
    root = repo_root or detect_repo_root()
    path = root / "deployment-yamls" / "serving-runtime.yaml.template"
    if not path.exists():
        raise FileNotFoundError(f"ServingRuntime template missing: {path}")
    return path.read_text(encoding="utf-8")


def render_serving_runtime(
    *,
    template_text: str,
    namespace: str,
    serving_runtime_name: str,
    vllm_runtime_image: str,
    model_format_name: str,
    display_name: str | None = None,
) -> str:
    """Replace placeholders in serving-runtime.yaml.template (KServe vLLM runtime)."""
    dn = (display_name or serving_runtime_name).strip() or serving_runtime_name
    repl = {
        "__NAMESPACE__": namespace.strip(),
        "__SERVING_RUNTIME_NAME__": serving_runtime_name.strip(),
        "__SERVING_RUNTIME_DISPLAY_NAME__": dn,
        "__VLLM_RUNTIME_IMAGE__": vllm_runtime_image.strip(),
        "__MODEL_FORMAT_NAME__": model_format_name.strip(),
    }
    text = template_text
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def validate_yaml_document(text: str) -> dict:
    docs = list(yaml.safe_load_all(text))
    if len(docs) != 1 or docs[0] is None:
        raise ValueError("Expected exactly one YAML document")
    return docs[0]


__all__ = [
    "sanitize_k8s_name",
    "normalize_dockerconfig_b64",
    "build_registry_secret_yaml",
    "render_inference_service",
    "load_inference_template",
    "load_serving_runtime_template",
    "render_serving_runtime",
    "validate_yaml_document",
    "pick_cpu_memory",
    "bump_memory_quantity",
    "halve_max_model_len_args",
    "format_gpu_limits_line",
]
