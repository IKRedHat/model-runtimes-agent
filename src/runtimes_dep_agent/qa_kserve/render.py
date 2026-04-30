"""Render KServe manifests from deployment-yamls templates."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import yaml

from ..utils.path_utils import detect_repo_root

_K8S_NAME_MAX = 63


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
    If the user passed raw JSON, base64-encode once. If already base64, return as-is
    (after stripping whitespace).
    """
    t = raw.strip()
    if t.startswith("{"):
        return base64.b64encode(t.encode("utf-8")).decode("ascii")
    return t


def build_registry_secret_yaml(
    *,
    registry_host: str,
    dockerconfigjson_b64: str,
    secret_name: str,
    namespace: str,
) -> str:
    """Build a valid kubernetes.io/dockerconfigjson Secret manifest."""
    obj = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": secret_name,
            "namespace": namespace,
            "annotations": {"registry.host/openshift": registry_host},
        },
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": dockerconfigjson_b64.strip()},
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


def pick_cpu_memory(
    *,
    required_vram_gb: float | None,
    heal_bump: int,
) -> tuple[str, str, str, str]:
    """
    Return cpu_request, memory_request, cpu_limit, memory_limit as Kubernetes quantities.
    heal_bump increases memory tiers on retry.
    """
    base_mem = 8
    if required_vram_gb is not None and required_vram_gb > 0:
        base_mem = max(base_mem, int(required_vram_gb * 1.25) + 2 + heal_bump * 4)
    else:
        base_mem = base_mem + heal_bump * 4

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
    "validate_yaml_document",
    "pick_cpu_memory",
    "bump_memory_quantity",
    "halve_max_model_len_args",
]
