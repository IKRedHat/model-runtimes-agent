"""
Deterministic deployability from models_info.json + gpu_info.txt.

GPU SKU and architecture are inferred only from GPU Product / provider lines,
never from VRAM capacity alone (e.g. 80 GB does not imply A100).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tensor parallel (weights-only lower bound)
# ---------------------------------------------------------------------------


def min_tensor_parallel_for_weights(estimated_weight_gb: float, per_gpu_mem_gb: float) -> int:
    """Minimum vLLM tensor parallel size to shard model weights across GPUs."""
    if per_gpu_mem_gb <= 0 or estimated_weight_gb <= 0:
        return 1
    return max(1, math.ceil(estimated_weight_gb / per_gpu_mem_gb))


# ---------------------------------------------------------------------------
# GPU inventory (parse gpu_info.txt)
# ---------------------------------------------------------------------------


@dataclass
class GpuInventory:
    """Parsed cluster accelerator snapshot from gpu_info-style text."""

    allocatable_gpus: int = 0
    per_gpu_mem_gb: float | None = None
    gpu_product_raw: str = ""
    gpu_provider: str = ""
    accelerator_family: str = "unknown"
    instance_type: str = ""
    source_lines: list[str] = field(default_factory=list)


_VOLTA = frozenset({"v100"})
_TURING = frozenset({"t4", "rtx", "turing"})
_AMPERE = frozenset({"a100", "a30", "a10", "a16", "ampere"})
_ADA = frozenset({"l4", "l40", "l40s", "ada", "rtx 40"})
_HOPPER = frozenset({"h100", "h200", "h800", "hopper"})


def _norm_product(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def infer_accelerator_family(gpu_product_line: str, gpu_provider: str = "") -> str:
    """
    Map GPU Product string to a coarse family for quantization rules.
    Uses product name tokens only, not memory capacity.
    """
    combined = _norm_product(gpu_product_line + " " + gpu_provider)
    if not combined.strip():
        return "unknown"

    # AMD / Intel-style before NVIDIA substring checks
    if any(x in combined for x in ("mi300", "mi325", "mi250", "instinct")):
        return "amd_gpu"
    if (
        ("intel" in combined and "xe" in combined)
        or "max gpu" in combined
        or " flex " in f" {combined} "
    ):
        return "intel_gpu"

    tokens = set(re.findall(r"[a-z][a-z0-9]*", combined))

    if tokens & _HOPPER or "h100" in combined or "h200" in combined:
        return "hopper"
    if tokens & _ADA or "l40" in combined or "l4 " in combined:
        return "ada"
    if tokens & _AMPERE or "a100" in combined or "a30" in combined:
        return "ampere"
    if tokens & _TURING or "t4" in combined:
        return "turing"
    if tokens & _VOLTA or "v100" in combined:
        return "volta"

    if "nvidia" in combined or "geforce" in combined:
        return "nvidia_unknown"

    return "unknown"


def parse_gpu_inventory(gpu_text: str) -> GpuInventory:
    """
    Parse bullet-style gpu_info.txt: sum Allocatable GPUs, prefer explicit Per-GPU Memory
    over extracting GB from product marketing names.
    """
    inv = GpuInventory()
    if not gpu_text or not gpu_text.strip():
        return inv

    inv.source_lines = [ln.strip() for ln in gpu_text.strip().splitlines() if ln.strip()]
    per_explicit: float | None = None
    per_from_product: float | None = None

    for line in inv.source_lines:
        low = line.lower()
        if "allocatable gpus" in low:
            m = re.search(r"(\d+)", line)
            if m:
                inv.allocatable_gpus += int(m.group(1))
        if "gpu provider" in low and ":" in line:
            inv.gpu_provider = line.split(":", 1)[1].strip()
        if "instance type" in low and ":" in line:
            inv.instance_type = line.split(":", 1)[1].strip()
        if "gpu product" in low and ":" in line:
            inv.gpu_product_raw = line.split(":", 1)[1].strip()
            m = re.search(r"(\d+(?:\.\d+)?)\s*gb", line, re.IGNORECASE)
            if m:
                try:
                    per_from_product = float(m.group(1))
                except ValueError:
                    pass
        if "per-gpu memory" in low or "per gpu memory" in low:
            m = re.search(r"(\d+(?:\.\d+)?)", line)
            if m:
                try:
                    per_explicit = float(m.group(1))
                except ValueError:
                    pass

    inv.per_gpu_mem_gb = per_explicit if per_explicit is not None else per_from_product
    inv.accelerator_family = infer_accelerator_family(inv.gpu_product_raw, inv.gpu_provider)
    return inv


# ---------------------------------------------------------------------------
# Model quantization inference (rule-based)
# ---------------------------------------------------------------------------

_FP8_NAME = re.compile(r"[-_]fp8\b|fp8[-_:]|/fp8|\.fp8\b", re.IGNORECASE)
_AWQ = re.compile(r"\bawq\b", re.IGNORECASE)
_GPTQ = re.compile(r"\bgptq\b", re.IGNORECASE)
_GGUF = re.compile(r"\bgguf\b", re.IGNORECASE)
_BNB = re.compile(r"bitsandbytes|bnb\b", re.IGNORECASE)
_W4A = re.compile(r"w\d+a\d+", re.IGNORECASE)


def infer_quantization_kind(model_key: str, info: dict[str, Any]) -> str | None:
    """
    Return a coarse label for compatibility checks, or None if unknown / generic FP16.
    """
    name = (info.get("image") or "") + " " + (info.get("model_name") or model_key)
    nlow = name.lower()

    if _FP8_NAME.search(nlow) or re.search(r"\bfp8\b", nlow):
        return "fp8"
    if _AWQ.search(nlow):
        return "awq"
    if _GPTQ.search(nlow):
        return "gptq"
    if _GGUF.search(nlow):
        return "gguf"
    if _BNB.search(nlow):
        return "bitsandbytes"
    if _W4A.search(nlow):
        return "w4a16_w8a8_style"

    qb = info.get("quantization_bits")
    if qb is not None:
        try:
            q = int(qb)
        except (TypeError, ValueError):
            q = None
        else:
            if q == 8:
                # fp8 in name already handled; else treat as int8-style
                if "fp8" not in nlow:
                    return "int8"
            if q == 4:
                return "w4"

    return None


def _family_supports_fp8(family: str) -> bool:
    return family in ("hopper", "ada", "amd_gpu")


def _family_supports_awq(family: str) -> bool:
    return family in ("turing", "ampere", "ada", "hopper", "intel_gpu", "nvidia_unknown")


def _family_supports_gptq(family: str) -> bool:
    return family in ("volta", "turing", "ampere", "ada", "hopper", "intel_gpu", "nvidia_unknown")


def quantization_deployable(
    kind: str | None,
    family: str,
    gpu_product_raw: str,
) -> tuple[bool, str]:
    """
    Apply vLLM-style compatibility (subset of decision specialist matrix).
    If kind is None, allow (unknown quant → capacity-only path).
    """
    if kind is None:
        return True, "No specific quantization kernel inferred from name/metadata; capacity checks apply."

    gp = gpu_product_raw or "(GPU Product not parsed)"

    if kind == "fp8":
        if _family_supports_fp8(family):
            return True, f"FP8 kernels are supported on parsed accelerator family '{family}' (GPU Product: {gp})."
        return (
            False,
            f"FP8 (W8A8) is not supported on accelerator family '{family}' (GPU Product: {gp}); "
            "needs Ada, Hopper, or AMD MI-class GPU.",
        )

    if kind == "awq":
        if family == "amd_gpu":
            return False, f"AWQ is not supported on AMD GPU (GPU Product: {gp})."
        if _family_supports_awq(family):
            return True, f"AWQ supported on '{family}' (GPU Product: {gp})."
        return False, f"AWQ not supported on accelerator family '{family}' (GPU Product: {gp})."

    if kind == "gptq":
        if family == "amd_gpu":
            return False, f"GPTQ is not supported on AMD GPU (GPU Product: {gp})."
        if _family_supports_gptq(family):
            return True, f"GPTQ supported on '{family}' (GPU Product: {gp})."
        return False, f"GPTQ not supported on accelerator family '{family}' (GPU Product: {gp})."

    if kind in ("gguf", "bitsandbytes"):
        if family == "amd_gpu":
            return False, f"{kind} not supported on AMD GPU per compatibility matrix (GPU Product: {gp})."
        if family in ("intel_gpu",):
            return False, f"{kind} not supported on Intel GPU (GPU Product: {gp})."
        return True, f"{kind} supported on NVIDIA-class accelerators (family '{family}', GPU Product: {gp})."

    if kind == "int8":
        if family in ("volta", "amd_gpu", "intel_gpu"):
            return False, f"INT8 W8A8 path not supported on family '{family}' (GPU Product: {gp})."
        return True, f"INT8-style quantization compatible with family '{family}' (GPU Product: {gp})."

    if kind == "w4a16_w8a8_style":
        return True, "Name suggests W4A16/W8A8-style quantization; treat as compatible pending runtime (verify AWQ/GPTQ paths)."

    if kind == "w4":
        return True, "4-bit style weights; verify AWQ/GPTQ runtime path separately."

    return True, f"Quantization kind '{kind}' not fully classified; not blocking on quantization alone."


def _weight_proxy_gb(info: dict[str, Any]) -> tuple[float | None, str]:
    rv = info.get("required_vram_gb")
    if rv is not None:
        try:
            return float(rv), "required_vram_gb"
        except (TypeError, ValueError):
            pass
    ms = info.get("model_size_gb")
    if ms is not None:
        try:
            return (
                float(ms),
                "model_size_gb (weights-only proxy when required_vram_gb missing; reserve KV cache separately)",
            )
        except (TypeError, ValueError):
            pass
    return None, ""


def evaluate_model_row(
    model_name: str,
    info: dict[str, Any],
    inv: GpuInventory,
) -> dict[str, Any]:
    """Single matrix row: model_name, deployable, reason."""
    reasons: list[str] = []
    quant = infer_quantization_kind(model_name, info)
    q_ok, q_msg = quantization_deployable(quant, inv.accelerator_family, inv.gpu_product_raw)
    reasons.append(q_msg)

    if not q_ok:
        return {
            "model_name": model_name,
            "deployable": False,
            "reason": " ".join(reasons),
        }

    if inv.allocatable_gpus <= 0:
        return {
            "model_name": model_name,
            "deployable": False,
            "reason": (
                f"{q_msg} Cluster reports no allocatable GPUs (gpu_info). "
                f"GPU Product line: {inv.gpu_product_raw or 'n/a'}."
            ),
        }

    if inv.per_gpu_mem_gb is None or inv.per_gpu_mem_gb <= 0:
        return {
            "model_name": model_name,
            "deployable": False,
            "reason": (
                f"{q_msg} Cannot compute minimum tensor parallel: per-GPU memory not parsed from gpu_info.txt "
                f"(GPU Product: {inv.gpu_product_raw or 'n/a'})."
            ),
        }

    wgb, w_src = _weight_proxy_gb(info)
    if wgb is None:
        return {
            "model_name": model_name,
            "deployable": False,
            "reason": (
                f"{q_msg} No weight or VRAM proxy: set required_vram_gb or ensure model_size_gb in models_info."
            ),
        }

    min_tp = min_tensor_parallel_for_weights(wgb, inv.per_gpu_mem_gb)
    cap_ok = inv.allocatable_gpus >= min_tp

    cap_detail = (
        f"Minimum vLLM tensor-parallel-size ≈ {min_tp} from ceil({wgb} GB / {inv.per_gpu_mem_gb} GB per GPU) "
        f"using {w_src}; allocatable GPUs = {inv.allocatable_gpus}. "
        "(Weights-only; tune max-model-len for KV cache.)"
    )
    reasons.append(cap_detail)

    if not cap_ok:
        return {
            "model_name": model_name,
            "deployable": False,
            "reason": " ".join(reasons),
        }

    ok_reason = (
        f"Deployable: {q_msg} "
        f"{cap_detail} "
        f"Accelerator inferred from GPU Product as '{inv.accelerator_family}' (not from VRAM size). "
        f"Recommend --tensor-parallel-size={min_tp} and gpu_count={min_tp}."
    )
    return {"model_name": model_name, "deployable": True, "reason": ok_reason}


def compute_deployment_matrix(
    models_info: dict[str, Any],
    gpu_text: str,
) -> list[dict[str, Any]]:
    """Return deployment matrix rows for every entry in models_info."""
    inv = parse_gpu_inventory(gpu_text)
    rows: list[dict[str, Any]] = []
    for name, info in models_info.items():
        if not isinstance(info, dict):
            continue
        rows.append(evaluate_model_row(str(name), info, inv))
    return rows


def compute_deployment_matrix_from_paths(
    models_info_path: Path,
    gpu_info_path: Path,
) -> list[dict[str, Any]]:
    """Load JSON + gpu file from disk and compute matrix."""
    if not models_info_path.exists():
        return []
    try:
        data = json.loads(models_info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    gpu_text = ""
    if gpu_info_path.exists():
        try:
            gpu_text = gpu_info_path.read_text(encoding="utf-8")
        except OSError:
            gpu_text = ""
    return compute_deployment_matrix(data, gpu_text)


__all__ = [
    "GpuInventory",
    "compute_deployment_matrix",
    "compute_deployment_matrix_from_paths",
    "evaluate_model_row",
    "infer_accelerator_family",
    "infer_quantization_kind",
    "min_tensor_parallel_for_weights",
    "parse_gpu_inventory",
    "quantization_deployable",
]
