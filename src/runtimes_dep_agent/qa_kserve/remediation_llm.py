"""LLM-driven remediation proposals from logs/events/pod status (JSON-only)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from kubernetes.utils.quantity import parse_quantity
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .constants import max_gpu_allowed

logger = logging.getLogger(__name__)

# Conservative Kubernetes-style resource.Quantity: digits + optional fraction,
# optional suffix (CPU n/u/m or binary SI Ki..Ei or decimal SI K..P).
_K8S_QTY_WHITELIST = re.compile(
    r"^\d+(\.\d+)?(?:[num]|(?:[KMGTPE])i|[KMGTPE])?$",
    re.IGNORECASE,
)


@dataclass
class RemediationPlan:
    """Validated proposal after LLM + sanity checks."""

    summary: str
    serving_arguments: list[str]
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str
    gpu_count: int


def _clamp_gpu(n: int, cap: int) -> int:
    return max(0, min(n, cap))


def _validate_quantity(value: str) -> bool:
    """True if value is a plausible Kubernetes resource quantity (CPU/memory)."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or len(s) > 32:
        return False
    if not s.isascii() or any(ord(c) < 32 for c in s):
        return False
    if _K8S_QTY_WHITELIST.fullmatch(s) is None:
        return False
    try:
        parse_quantity(s)
    except Exception:
        return False
    return True


def parse_llm_json(content: str) -> dict[str, Any] | None:
    """Extract JSON object from model output (raw or fenced)."""
    raw = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        text = raw
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def validate_and_build_plan(
    raw: dict[str, Any],
    *,
    max_gpu: int,
    fallback_args: list[str],
    fallback_cpu_req: str,
    fallback_mem_req: str,
    fallback_cpu_lim: str,
    fallback_mem_lim: str,
    fallback_gpu: int,
) -> RemediationPlan | None:
    """Map LLM JSON to RemediationPlan or return None if unusable."""
    summary = str(raw.get("summary") or "remediation")[:2000]
    args = raw.get("serving_arguments")
    if not isinstance(args, list):
        args = fallback_args
    else:
        args = [str(a) for a in args if isinstance(a, (str, int, float))]

    res = raw.get("resources")
    if not isinstance(res, dict):
        res = {}

    def _coerce_qty(val: Any, fb: str) -> str:
        cand = str(val).strip() if val is not None else fb
        if _validate_quantity(cand):
            return cand
        logger.warning("Invalid quantity %r — using fallback %r", val, fb)
        return fb

    cpu_req = _coerce_qty(res.get("cpu_request"), fallback_cpu_req)
    mem_req = _coerce_qty(res.get("memory_request"), fallback_mem_req)
    cpu_lim = _coerce_qty(res.get("cpu_limit"), fallback_cpu_lim)
    mem_lim = _coerce_qty(res.get("memory_limit"), fallback_mem_lim)
    gpu_raw = res.get("gpu_count", fallback_gpu)

    try:
        gpu_n = int(gpu_raw)
    except (TypeError, ValueError):
        gpu_n = fallback_gpu
    gpu_n = _clamp_gpu(gpu_n, max_gpu)

    if not args:
        args = list(fallback_args)

    return RemediationPlan(
        summary=summary,
        serving_arguments=args,
        cpu_request=cpu_req,
        memory_request=mem_req,
        cpu_limit=cpu_lim,
        memory_limit=mem_lim,
        gpu_count=gpu_n,
    )


def propose_remediation(
    llm: BaseChatModel,
    *,
    context: dict[str, Any],
    fallback_args: list[str],
    fallback_cpu_req: str,
    fallback_mem_req: str,
    fallback_cpu_lim: str,
    fallback_mem_lim: str,
    fallback_gpu: int,
) -> RemediationPlan | None:
    """
    Ask the LLM for a single JSON remediation plan. Returns None if parse/validate fails.

    `context` should include string keys: model_name, isvc_name, wait_detail, pod_json_excerpt,
    events_tail, logs_storage_initializer, logs_kserve_container, current_args_json,
    current_resources_json, gpu_provider, max_gpu_allowed.
    """
    sys_msg = (
        "You are an OpenShift KServe / vLLM deployment engineer. "
        "Given failure context (pod status, events, container logs), output ONLY a single JSON object "
        "with this exact shape (no markdown outside JSON):\n"
        "{\n"
        '  "summary": "short root cause for operators",\n'
        '  "serving_arguments": ["--flag=value", ...],\n'
        '  "resources": {\n'
        '    "cpu_request": "2",\n'
        '    "memory_request": "16Gi",\n'
        '    "cpu_limit": "8",\n'
        '    "memory_limit": "32Gi",\n'
        '    "gpu_count": 1\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- serving_arguments is the FULL replacement list for vLLM args (include all flags needed).\n"
        "- Use quantities compatible with Kubernetes (e.g. 16Gi, 500m, 2).\n"
        "- gpu_count must be an integer between 0 and the provided max_gpu_allowed.\n"
        "- If failure is registry/auth/image pull, still output safer smaller resources but note in summary; "
        "  outside tooling may skip retries for image pull.\n"
        "- Do not include secrets or kubeconfig.\n"
    )
    user_parts = [
        f"model_name: {context.get('model_name')}",
        f"inferenceservice: {context.get('isvc_name')}",
        f"wait_detail: {context.get('wait_detail')}",
        f"gpu_provider: {context.get('gpu_provider')}",
        f"max_gpu_allowed: {context.get('max_gpu_allowed')}",
        f"current_args_json: {context.get('current_args_json')}",
        f"current_resources_json: {context.get('current_resources_json')}",
        "--- pod_json_excerpt ---\n" + str(context.get("pod_json_excerpt") or "")[:8000],
        "--- events_tail ---\n" + str(context.get("events_tail") or "")[:6000],
        "--- logs storage-initializer ---\n" + str(context.get("logs_storage_initializer") or "")[:6000],
        "--- logs kserve-container ---\n" + str(context.get("logs_kserve_container") or "")[:6000],
    ]
    human_msg = "\n\n".join(user_parts)

    try:
        model = llm
        bind = getattr(llm, "bind", None)
        if callable(bind):
            try:
                model = bind(temperature=0)
            except TypeError:
                model = llm
        out = model.invoke(
            [SystemMessage(content=sys_msg), HumanMessage(content=human_msg)],
        )
        content = getattr(out, "content", out)
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        raw = parse_llm_json(str(content))
        if not raw:
            logger.warning("LLM remediation returned no parseable JSON")
            return None

        plan = validate_and_build_plan(
            raw,
            max_gpu=int(context.get("max_gpu_allowed") or max_gpu_allowed()),
            fallback_args=list(fallback_args),
            fallback_cpu_req=fallback_cpu_req,
            fallback_mem_req=fallback_mem_req,
            fallback_cpu_lim=fallback_cpu_lim,
            fallback_mem_lim=fallback_mem_lim,
            fallback_gpu=fallback_gpu,
        )
        return plan
    except Exception:
        logger.exception("LLM remediation failed")
        return None


__all__ = [
    "RemediationPlan",
    "propose_remediation",
    "validate_and_build_plan",
    "parse_llm_json",
]
