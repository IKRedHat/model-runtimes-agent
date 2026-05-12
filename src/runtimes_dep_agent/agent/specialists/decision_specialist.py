"""Decision specialist that compares model requirements with cluster capacity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from . import SpecialistSpec
from ...utils.path_utils import detect_repo_root
from ...validators.deployability_engine import (
    compute_deployment_matrix,
    min_tensor_parallel_for_weights,
    parse_gpu_inventory,
)

logger = logging.getLogger(__name__)


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = detect_repo_root([CURRENT_FILE])
INFO_DIR = REPO_ROOT / "info"
GPU_INFO_DEFAULT = INFO_DIR / "gpu_info.txt"
DEPLOYMENT_INFO_DEFAULT = INFO_DIR / "deployment_info.txt"


def build_decision_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
    info_dir: Path | None = None,
) -> SpecialistSpec:
    """Create the decision specialist that determines deployment feasibility."""
    effective_info_dir = info_dir if info_dir is not None else INFO_DIR

    @tool
    def describe_preloaded_requirements() -> str:
        """Return the preloaded model requirements as JSON."""
        return json.dumps(precomputed_requirements, indent=2)
    
    @tool
    def assess_deployment_fit(file_path: str | None = None) -> str:
        """Evaluate whether cached models fit on the cluster GPUs (optionally override GPU info path)."""
        if not precomputed_requirements:
            return "Deployment Fit Analysis:\n- No preloaded model requirements available."

        gpu_file = Path(file_path) if file_path else (effective_info_dir / "gpu_info.txt")
        if not gpu_file.exists():
            return f"Deployment Fit Analysis:\n- GPU info file not found at {gpu_file}."

        try:
            gpu_text = gpu_file.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Deployment Fit Analysis:\n- Error reading GPU info file ({gpu_file}): {exc}"

        inv = parse_gpu_inventory(gpu_text)
        total_gpus = inv.allocatable_gpus
        per_gpu_mem = inv.per_gpu_mem_gb

        per_model_lines = []
        total_required = 0
        for name, info in precomputed_requirements.items():
            required_vram = info.get("required_vram_gb")
            model_size_gb = info.get("model_size_gb")
            weight_proxy: float | None = None
            proxy_note = ""

            if required_vram is not None:
                try:
                    weight_proxy = float(required_vram)
                except (TypeError, ValueError):
                    weight_proxy = None
            elif model_size_gb is not None:
                try:
                    weight_proxy = float(model_size_gb)
                    proxy_note = (
                        f" (catalog `required_vram_gb` missing; using `model_size_gb`={weight_proxy} GB "
                        "as a weights-only proxy for tensor-parallel sizing — add KV-cache / runtime headroom in ops)"
                    )
                except (TypeError, ValueError):
                    weight_proxy = None

            if weight_proxy is not None and per_gpu_mem:
                min_tp = min_tensor_parallel_for_weights(weight_proxy, per_gpu_mem)
                total_required += min_tp
                per_model_lines.append(
                    f"- {name}: minimum vLLM `--tensor-parallel-size` ≈ {min_tp} "
                    f"(ceil({weight_proxy} / {per_gpu_mem}) for weight sharding across ~{per_gpu_mem} GB GPUs)"
                    f"{proxy_note}. "
                    f"Prefer this **minimal** TP — do **not** set tensor parallel to the full cluster size ({total_gpus}) "
                    "unless latency/throughput goals require it; extra GPUs can stay unused or run separate replicas."
                )
            elif weight_proxy is not None:
                per_model_lines.append(
                    f"- {name}: weight proxy {weight_proxy} GB available{proxy_note}, "
                    "but per-GPU memory is unknown — cannot compute minimum tensor parallel."
                )
            elif required_vram:
                per_model_lines.append(
                    f"- {name}: requires {required_vram} GB VRAM but per-GPU memory is unknown."
                )
            else:
                per_model_lines.append(f"- {name}: VRAM requirement could not be inferred (no model_size_gb either).")

        comparison = "Insufficient data to compare cluster capacity with model needs."
        if per_gpu_mem and total_required:
            if total_gpus >= total_required:
                comparison = (
                    f"Cluster GPUs available ({total_gpus}) meet or exceed the inferred minimum for "
                    f"weight sharding ({total_required} GPU(s) at `--tensor-parallel-size`={total_required} for this catalog)."
                )
            else:
                comparison = (
                    f"Cluster GPUs available ({total_gpus}) are below the inferred minimum tensor-parallel need "
                    f"({total_required})."
                )

        per_model_report = "\n".join(per_model_lines) if per_model_lines else "- No models found."
        return (
            "Deployment Fit Analysis:\n"
            f"- Source GPU file: {gpu_file}\n"
            f"- Total GPUs available: {total_gpus}\n"
            f"- Per-GPU memory (parsed): {per_gpu_mem or 'unknown'} GB\n"
            f"- GPU Product (from file): {inv.gpu_product_raw or 'unknown'}\n"
            f"- Accelerator family (inferred from GPU Product / provider, not from VRAM size): {inv.accelerator_family}\n"
            "- Per-model breakdown:\n"
            f"{per_model_report}\n"
            f"- Comparison: {comparison}"
        )
    
    @tool
    def deployability_decision(deployment_matrix_json: str = "") -> str:
        """
        Write info/deployment_matrix.json using deterministic rules (deployability engine).

        Reads models_info.json + gpu_info.txt, infers GPU SKU only from GPU Product lines,
        evaluates quantization compatibility and minimum tensor parallel size. The
        deployment_matrix_json argument is ignored (legacy agents may pass "{}").
        """
        json_path = effective_info_dir / "deployment_matrix.json"

        models_data: dict = {}
        mi_path = effective_info_dir / "models_info.json"
        try:
            if mi_path.exists():
                raw = json.loads(mi_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    models_data = raw
        except (OSError, json.JSONDecodeError):
            models_data = {}
        if not models_data and precomputed_requirements:
            models_data = precomputed_requirements

        gpu_text = ""
        try:
            gf = effective_info_dir / "gpu_info.txt"
            if gf.exists():
                gpu_text = gf.read_text(encoding="utf-8")
        except OSError:
            pass

        reconciled = compute_deployment_matrix(models_data, gpu_text)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reconciled, f, indent=2)

        deployable_models_r = []
        non_deployable_models_r = []
        for entry in reconciled:
            model_name = entry.get("model_name", "unknown")
            if entry.get("deployable", False):
                deployable_models_r.append(f"- {model_name}: Deployable")
            else:
                non_deployable_models_r.append(
                    f"- {model_name}: Not Deployable ({entry.get('reason', 'No reason provided.')})"
                )
        deployable_report = "\n".join(deployable_models_r) if deployable_models_r else "- None"
        non_deployable_report = (
            "\n".join(non_deployable_models_r) if non_deployable_models_r else "- None"
        )

        return (
            "Deployability Decision Report:\n"
            "Deployable Models:\n"
            f"{deployable_report}\n\n"
            "Non-Deployable Models:\n"
            f"{non_deployable_report}"
        )
        


    prompt = """
        You are a deployment decision specialist.

        CRITICAL: Deployable vs non-deployable is computed deterministically by the tool
        deployability_decision() from models_info.json + gpu_info.txt (GPU Product line,
        not VRAM-inferred SKU). Do NOT contradict the tool output or invent matrix rows.

        You MUST ALWAYS call these tools in order:
        1. describe_preloaded_requirements() - structured model metadata.
        2. assess_deployment_fit() - GPU counts, per-GPU memory, accelerator family,
           minimum tensor-parallel hints per model.
        3. deployability_decision() - writes info/deployment_matrix.json (argument ignored;
           pass "{}"). Matrix rows are authoritative.

        Your narrative job after calling tools:
        - Summarize deployability_decision output for the human report.
        - For OPTIMIZED_SERVING_ARGUMENTS_JSON: align `--tensor-parallel-size` and
          `gpu_count` with the **minimum** TP from assess_deployment_fit (not total GPUs).
          Baseline args when missing:
            --uvicorn-log-level=info
            --trust-remote-code
            --max-model-len=2048
        - `supported_arch` in models_info is **CPU image arch**, not GPU SKU.

        Never infer Ampere vs Hopper from "80 GB" VRAM alone — only gpu_info GPU Product.
        """


    agent = create_agent(
        llm,
        tools=[assess_deployment_fit, describe_preloaded_requirements, deployability_decision],
        system_prompt=prompt,
    )

    @tool
    def analyze_deployment_decision(request: str) -> str:
        """Delegate deployment fit decisions to the decision specialist."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        output_text = extract_text(result)
        
        # Save deployment decision output to info/deployment_info.txt
        deployment_info_path = effective_info_dir / "deployment_info.txt"
        try:
            effective_info_dir.mkdir(parents=True, exist_ok=True)
            with open(deployment_info_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
        except Exception as e:
            # Log error but don't fail the tool
            logger.error(f"Failed to save deployment info to {deployment_info_path}: {e}")
        
        return output_text

    analyze_deployment_decision.name = "analyze_deployment_decision"

    return SpecialistSpec(
        name="decision_specialist",
        agent=agent,
        tool=analyze_deployment_decision,
    )


__all__ = ["build_decision_specialist", "min_tensor_parallel_for_weights"]
